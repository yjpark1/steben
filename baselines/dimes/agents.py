import datetime
import ray
from ray.rllib.utils.compression import pack, unpack, unpack_if_needed

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from collections import deque
import numpy as np
import networkx as nx
import tqdm
import os

from stpgen.datasets.synthetic.instance import (MAX_EDGE_COST,
                                                 STPInstance_erdos_renyi,
                                                 STPInstance_grid,
                                                 STPInstance_regular,
                                                 STPInstance_watts_strogatz)
from stpgen.solvers.heuristic import TwoApproximation, remove_inessentials
from baselines.cherrypick.env import STPEnvironment
from baselines.dimes.env import STPEdgeEnvironment, STPTorchEnvironment
from baselines.dimes_tsp.inc.utils import torch_add_grad
from baselines.cherrypick.networks import GraphEmbedding
from baselines.dimes.worker import Worker
from baselines.dimes.util import new_env, get_instance_sampler_from_graph_type

class DimesAgent:
    def __init__(self, network_cls, args):
        self.network = network_cls(args).to(args.device)
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        ray.init(ignore_reinit_error=True)

        self.batch_size = args.tr_batch_size
        self.args = args
        self.steps = 0
        self.tr_gamma = args.tr_gamma
        self.neg_inf = torch.full((1,), -np.inf, dtype=torch.float32, device=self.device)
        self.tr_envs = [
            new_env(args.n_nodes, args.seed + idx, args, args.graph_type, args.cost_type, self.device)
            for idx in range(args.tr_batch_size)
        ]
        worker_cls = Worker.as_remote(num_cpus=1)
        self.spl_envs = [
            new_env(args.n_nodes, args.seed + len(self.tr_envs), args, args.graph_type, args.cost_type, self.device)
        ]
        pydev_port = os.environ.get('PYDEV_PORT', None)
        self.workers = [
            worker_cls.remote(network_cls, args, args.seed + idx + len(self.tr_envs) + 1, pydev_port=pydev_port) for idx in range(args.tr_inner_sample_size)
        ]

    @torch.no_grad()
    def get_heatmap_from_state(self, state):
        instance_emb, graph_emb = self._state_to_emb(state, reduce_dim=True)
        heatmap = self.network.par_net(graph_emb)
        return instance_emb, graph_emb, heatmap

    @torch.no_grad()
    def get_action_prob_from_state_and_heatmap(self, state, heatmap, select, temperature=1.0):
        par_e = torch.where(state.available_edges, heatmap, self.neg_inf)
        logits_ = par_e - torch.max(par_e)
        par_exp_ = (logits_ / temperature).exp()
        p_denom_ = torch.sum(par_exp_)
        probs_t = par_exp_ / p_denom_
        if select == 'max':
            action = torch.argmax(probs_t)
        elif select == 'gumbel':
            gumb = torch.empty_like(par_e).exponential_().log()
            action = torch.argmax(par_e - gumb)
        elif select == 'softmax':
            m = Categorical(probs_t)
            action = m.sample()
        else:
            raise ValueError
        return action, probs_t

    def get_action(self, state, heatmap=None, select='max', temperature=1.0):
        if heatmap is None:
            _, _, heatmap = self.get_heatmap_from_state(state)
        action, prob_t = self.get_action_prob_from_state_and_heatmap(state, heatmap, select, temperature)
        return action, heatmap, prob_t

    def process_single_state(self, state):
        num_vertices = state.graph.number_of_nodes()
        sv = self._create_binary_vector(state.partial_solution, num_vertices)
        tv = self._create_binary_vector(state.graph.graph['Terminals']['terminals'], num_vertices)
        xv = np.array([v for _, v in sorted(state.distance.items(), key=lambda x: x[0])])
        adj = nx.adjacency_matrix(state.graph).todense()

        sv = np.array([sv], dtype=np.float32)
        tv = np.array([tv], dtype=np.float32)
        xv = np.array([xv], dtype=np.float32)
        adj = np.array([adj], dtype=np.float32)

        features = (
            torch.from_numpy(sv).to(self.device),
            torch.from_numpy(tv).to(self.device),
            torch.from_numpy(xv).to(self.device),
            torch.from_numpy(adj).to(self.device)
        )

        return features

    @staticmethod
    def _create_binary_vector(indices, size):
        binary_vector = np.zeros(size, dtype=np.float32)
        # Set the specified indices to 1
        binary_vector[indices] = 1
        return binary_vector

    def _create_dimes_state(self, state):
        return state

    def get_two_approx_sol(self, graph):
        two_approx = TwoApproximation(graph)
        two_approx.solve()
        sol = two_approx.solution
        solved = graph.edge_subgraph(sol.edges).copy()
        c_huer = np.sum(list(map(lambda x: x[2]['cost'], solved.edges(data=True))))
        return solved, c_huer

    def train(self, writer):
        self.network.train()
        opt = self.args.outer_opt_fn(self.network.parameters())
        tbar = range(1, self.args.tr_outer_steps + 1)
        tbar = tqdm.tqdm(tbar)

        y_list = []
        c_list = []
        history = {
            'reward': [],
            'cost': [],
            'gap': [],
            'global_step': 0,
        }
        c_queue = deque(maxlen=min(3, self.args.tr_outer_steps))
        last_c_mean = float('inf')

        mean_gap = 0
        c_min = float('inf')
        for step in tbar:
            lst = [env.reset() for env in self.tr_envs]
            states, infos = list(map(list, zip(*lst)))
            instances = [env.instance for env in self.tr_envs]
            two_approx_sol = []
            two_approx_costs = []
            for env, instance in zip(self.tr_envs, instances):
                solved, c_huer = self.get_two_approx_sol(instance)
                c_huer /= env.max_edge_cost
                two_approx_costs.append(c_huer)
                two_approx_sol.append(solved)

            instance_embs, emb0_list = self._state_to_emb(states)
            emb0_batch = torch.cat(emb0_list, 0)
            emb0_grads = []

            phi_grad_lists = []
            for phi in self.network.par_net.trainables():
                phi_grad_lists.append([])


            for i, (x, graph, emb0) in enumerate(zip(instance_embs, states, emb0_list)):
                emb1, psi_net, ys, _, costs = self.stp_tune(emb0, self.network.par_net, graph, self.args.inner_opt_fn,
                                                     self.args.tr_inner_steps, self.args.tr_inner_sample_size,
                                                     greedy_size=None, verbose=False, plot=False)

                y1 = float(np.mean(ys))
                y_list.append(y1)
                last_c = float(np.mean(costs))
                # c_list.append(last_c)
                # c_queue.append(last_c)
                emb0_grad, phi_grads = self.net_approx_grads(emb1, psi_net, graph, sample_size=self.args.tr_inner_sample_size)
                emb0_grads.append(emb0_grad)
                for phi_grad_list, phi_grad in zip(phi_grad_lists, phi_grads):
                    phi_grad_list.append(phi_grad)

                # test_ys, test_costs = self.eval_instance_with_heatmap(graph.graph, pruning=bool(self.args.pruning))
                # y_this_step.append(float(test_ys))
                # c_this_step.append(float(test_costs))
                #
                # c_list.append(float(test_costs))
            # mean_this_step = float(np.mean(c_this_step))
            # c_queue.append(mean_this_step)

            opt.zero_grad()
            emb0_grads = torch.cat(emb0_grads, dim=0) / self.args.tr_batch_size
            emb0_batch.backward(emb0_grads.detach())

            for phi, phi_grad_list in zip(self.network.par_net.trainables(), phi_grad_lists):
                torch_add_grad(phi, torch.stack(phi_grad_list, dim=0).mean(dim=0).detach())

            opt.step()
            self.network.eval()

            c_this_step = []
            y_this_step = []
            gaps = []

            for eval_epi in range(100):
                temp_ys = []
                temp_cs = []
                env = self.spl_envs[0]

                state, info = env.reset()
                inst = env.instance.copy()

                solver = TwoApproximation(inst)
                solver.solve()
                solver.solution = env.instance.edge_subgraph(solver.solution.edges).copy()

                c_huer = np.sum(list(map(lambda x: x[2]['cost'], solver.solution.edges(data=True)))) / env.max_edge_cost
                temp_ys, temp_cs = self.eval_parallel_instance(inst, select=self.args.test_method, temperature=self.args.temperature)
                temp_ys = [float(t1.detach().cpu().numpy()) for t1 in temp_ys]
                # for ti in range(32):
                #     with torch.no_grad():
                #         test_ys, test_costs = self.eval_instance_with_heatmap(inst, pruning=bool(self.args.pruning))
                #         temp_ys.append(float(test_ys))
                #         temp_cs.append(float(test_costs))
                temp_y = max(temp_ys)
                temp_c = min(temp_cs)
                c_this_step.append(temp_c)
                y_this_step.append(temp_y)
                gaps.append(temp_c / c_huer)

            self.network.train()

            mean_gap = float(np.mean(gaps))
            tbar.set_description(f'step {step}, mean_gap: {mean_gap * 100:.1f}')

            avg_rewards = float(np.mean(y_this_step))
            avg_costs = float(np.mean(c_this_step))
            c_queue.append(avg_costs)
            last_c = mean_gap

            env_episodes = step * self.args.tr_batch_size * self.args.tr_inner_steps # * self.args.tr_inner_sample_size
            writer.add_scalar("performance/reward", avg_rewards, env_episodes)
            writer.add_scalar("performance/cost", avg_costs, env_episodes)
            writer.add_scalar("performance/gap", mean_gap, env_episodes)
            history['reward'].append(avg_rewards)
            history['cost'].append(avg_costs)
            history['gap'].append(mean_gap)
            history['global_step'] = step
            last_c_mean = np.mean(c_queue)

            # if len(c_queue) == c_queue.maxlen and last_c_mean > c_mean:
            if last_c < c_min:
                c_min = last_c
                c_mean = last_c_mean
                torch.save(self.network.state_dict(), f"{self.args.log_dir}/{self.args.filename_for_dqn_weights}.pth")
            torch.save(self.network.state_dict(), f"{self.args.log_dir}/{self.args.filename_for_dqn_weights}_{step}.pth")

        return history

    def net_approx_grads(self, emb, psi_net, graph, sample_size):
        emb = emb.detach().clone().requires_grad_()
        if emb.grad is not None:
            emb.grad.zero_()
        par = psi_net(emb)
        # degree = [e1[1] for e1 in list(graph.graph.degree)]
        _, par_grad, _ = self.stp_softmax_grad_par(par.detach(), sample_size, y_bl=None, instance=graph.graph)
        par_grad = par_grad.to(self.device)
        par.backward(par_grad)
        emb_grad = emb.grad.detach().clone()
        phi_grads = []
        for psi in psi_net.trainables():
            phi_grads.append(psi.grad.detach().clone())
        return emb_grad, phi_grads

    def _state_to_emb(self, states, reduce_dim=False):
        if not isinstance(states, list):
            states = [states]

        instance_emb = []
        graph_emb = []
        for state in states:
            s, t, x, adj = self.process_single_state(state)
            emb = self.network.graph_emb(s, t, x).squeeze(0)
            instance_emb.append(emb)

            emb2 = self.network.emb_net(emb, state.edge_list.transpose(0, 1), state.edge_cost.unsqueeze(-1))
            graph_emb.append(emb2)

        if reduce_dim and len(states) == 1:
            return instance_emb[0], graph_emb[0]

        return instance_emb, graph_emb

    def stp_tune(self, emb0, phi_net, graph, opt_fn, steps, sample_size, greedy_size,
                 verbose = True, plot = True, save_name = None):
        emb = emb0.detach().clone().requires_grad_()
        psi_net = phi_net.clone()
        psi_net.train()
        opt = opt_fn([emb, *psi_net.trainables()])
        tbar = range(1, steps + 1)
        y_means = []
        costs_means = []

        for t in tbar:
            opt.zero_grad()
            par = psi_net(emb)
            ys, par_grad, costs = self.stp_softmax_grad_par(par, sample_size, y_bl=None, instance=graph.graph)
            par_grad = par_grad.to(self.device)
            par.backward(par_grad)
            opt.step()
            y_means.append(ys.mean().item())
            costs_means.append(np.mean(costs).item())

        return emb, psi_net, y_means, [None], costs_means

    def stp_softmax_grad_par(self, par, sample_size, y_bl, instance, select='gumbel'):
        if sample_size > len(self.workers):
            sample_size = len(self.workers)
        par_cpu = par.detach().cpu()
        result = ray.get([
            worker.rollout.remote(par_cpu, pack(instance), select=select) for worker in self.workers[:sample_size]
        ])
        result = [
            unpack_if_needed(r1) for r1 in result
        ]
        cumulative_rewards, rewards, logits, available_edges, actions, costs = list(map(list, zip(*result)))

        cumulative_rewards = torch.cat(cumulative_rewards, -1)
        r_bl = torch.mean(cumulative_rewards)
        par_shape = par.shape
        grads = ray.get([
            worker.grad.remote(par_shape, r_bl) for worker in self.workers[:sample_size]
        ])
        grad = torch.sum(torch.stack(grads, dim=0), dim=0) / sample_size
        return cumulative_rewards, grad, costs

    def stp_softmax_grad_torch(self, par, sample_size, y_bl, instance, select='gumbel'):
        pass


    def eval_instance_with_heatmap(self, instance=None, heatmap=None, pruning=True, initial_vertex_idx=None, select='max', temperature=1.0):
        env = self.spl_envs[0]
        if instance is None:
            state, info = env.reset()
        else:
            state, info = env.reset(instance.copy(), initial_vertex_idx=initial_vertex_idx)
        total_reward = 0
        terminated = False
        actions = []
        while not terminated:
            with torch.no_grad():
                action, heatmap, prob_t = self.get_action(state, heatmap=heatmap, select=select, temperature=temperature)

            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            actions.append(action.reshape(1))
            # logging
            total_reward += reward
            if terminated:
                break

        if pruning:
            edge_list = env.edge_list
            edges = [tuple(edge_list[a1].squeeze().cpu().numpy()) for a1 in actions]
            new_graph = env.instance.edge_subgraph(edges).copy()
            new_graph.graph = env.instance.graph
            solution = remove_inessentials(new_graph, new_graph.graph['Terminals']['terminals'])

            c_huer = np.sum(list(map(lambda x: x[2]['cost'], solution.edges(data=True)))) / env.max_edge_cost
            return -c_huer, c_huer

        return float(total_reward.detach().cpu().numpy()), float(env.current_cost.detach().cpu().numpy())

    def eval_multiple_jobs_with_worker(self, jobs, worker_idx, status_worker):
        return self.workers[worker_idx].multiple_rollout.remote(jobs, status_worker)

    def eval_instance_with_worker(self, instance, worker_idx, select='max', temperature=1, pruning=False, initial_vertex_idx=None):
        terminals = instance.graph['Terminals']['terminals']
        state, info = self.spl_envs[0].reset(instance=instance)

        init_terminal_index = np.where(np.array(terminals) == state.partial_solution[-1].item())[0][0]
        instance_emb, graph_emb, heatmap = self.get_heatmap_from_state(state)
        initial_vertex_idx = init_terminal_index if initial_vertex_idx is None else initial_vertex_idx
        ray_obj = self.workers[worker_idx].rollout.remote(
            heatmap.detach().cpu(),
            instance=pack(instance),
            select=select,
            temperature=temperature,
            initial_vertex_idx=initial_vertex_idx,
            pruning=pruning
        )
        return ray_obj

    def eval_parallel_instance(self, instance, samples=32, select='max', temperature=1, pruning=False):
        state, info = self.spl_envs[0].reset(instance=instance)
        instance_emb, graph_emb, heatmap = self.get_heatmap_from_state(state)
        results = []
        terminals = instance.graph['Terminals']['terminals']
        base_idx = 0
        while samples > 0:
            if samples >= len(self.workers):
                num_worker = len(self.workers)
            else:
                num_worker = samples

            current_result = ray.get([
                self.workers[idx].rollout.remote(
                    heatmap.detach().cpu(), instance=pack(instance), select=select,
                    temperature=temperature,
                    pruning=pruning,
                    initial_vertex_idx=(base_idx + idx) % len(terminals))
                for idx in range(len(self.workers[:num_worker]))
            ])
            base_idx = (base_idx + num_worker) % len(terminals)

            results += current_result
            samples -= num_worker

        results = [
            unpack_if_needed(r1) for r1 in results
        ]
        cumulative_rewards, rewards, logits, available_edges, actions, costs = list(map(list, zip(*results)))
        return cumulative_rewards, costs

class DimesTorchAgent:
    def __init__(self, network_cls, args):
        self.network = network_cls(args).to(args.device)
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        self.args = args
        self.steps = 0
        self.tr_gamma = args.tr_gamma
        instance_sampler = get_instance_sampler_from_graph_type(args.n_nodes, args.seed, args.graph_type, args.cost_type)
        self.env = STPTorchEnvironment(instance_sampler, args.seed, device=self.device)

    def get_two_approx_sol(self, graph):
        two_approx = TwoApproximation(graph)
        two_approx.solve()
        sol = two_approx.solution
        solved = graph.edge_subgraph(sol.edges).copy()
        c_huer = np.sum(list(map(lambda x: x[2]['cost'], solved.edges(data=True))))
        return solved, c_huer


    @staticmethod
    def _create_binary_vector(indices, size):
        binary_vector = np.zeros(size, dtype=np.float32)
        # Set the specified indices to 1
        binary_vector[indices] = 1
        return binary_vector

    def process_single_state(self, state):
        num_vertices = state.graph.number_of_nodes()
        sv = self._create_binary_vector(state.partial_solution, num_vertices)
        tv = self._create_binary_vector(state.graph.graph['Terminals']['terminals'], num_vertices)
        xv = np.array([v for _, v in sorted(state.distance.items(), key=lambda x: x[0])])
        adj = nx.adjacency_matrix(state.graph).todense()

        sv = np.array([sv], dtype=np.float32)
        tv = np.array([tv], dtype=np.float32)
        xv = np.array([xv], dtype=np.float32)
        adj = np.array([adj], dtype=np.float32)

        features = (
            torch.from_numpy(sv).to(self.device),
            torch.from_numpy(tv).to(self.device),
            torch.from_numpy(xv).to(self.device),
            torch.from_numpy(adj).to(self.device)
        )

        return features

    def _state_to_emb(self, states, reduce_dim=False):
        if not isinstance(states, list):
            states = [states]

        instance_emb = []
        graph_emb = []
        for state in states:
            s, t, x, adj = self.process_single_state(state)
            emb = self.network.graph_emb(s, t, x).squeeze(0)
            instance_emb.append(emb)

            emb2 = self.network.emb_net(emb, state.edge_list.transpose(0, 1), state.edge_cost.unsqueeze(-1))
            graph_emb.append(emb2)

        if reduce_dim and len(states) == 1:
            return instance_emb[0], graph_emb[0]

        return instance_emb, graph_emb

    def train(self, writer):
        self.network.train()
        opt = self.args.outer_opt_fn(self.network.parameters())
        tbar = range(1, self.args.tr_outer_steps + 1)
        tbar = tqdm.tqdm(tbar)
        history = {
            'reward': [],
            'cost': [],
            'gap': [],
            'global_step': 0,
        }

        for step in tbar:
            heur_sol_batch = []
            heur_cost_batch = []
            states, infos = self.env.reset(num_instance=self.args.tr_batch_size)

            for tr_batch_idx in range(self.args.tr_batch_size):
                state = states[tr_batch_idx]
                heur_sol, heur_cost = self.get_two_approx_sol(state.graph)
                heur_sol_batch.append(heur_sol)
                heur_cost_batch.append(heur_cost)

            phi_grad_lists = []
            for phi in self.network.par_net.trainables():
                phi_grad_lists.append([])

            instance_embs, emb0_list = self._state_to_emb(states)

            for i, (x, graph, emb0) in enumerate(zip(instance_embs, states, emb0_list)):
                emb1, psi_net, ys, _, costs = self.stp_tune(emb0, self.network.par_net, graph, self.args.inner_opt_fn,
                                                            self.args.tr_inner_steps, self.args.tr_inner_sample_size)



        return {}

    def stp_tune(self, emb0, phi_net, graph, opt_fn, steps, sample_size):
        emb = emb0.detach().clone().requires_grad_()
        psi_net = phi_net.clone()
        psi_net.train()
        opt = opt_fn([emb, *psi_net.trainables()])
        tbar = range(1, steps + 1)
        y_means = []
        costs_means = []
        for t in tbar:
            opt.zero_grad()
            par = psi_net(emb)
            ys, par_grad, costs = self.stp_softmax_grad_par(par, sample_size, y_bl=None, instance=graph.graph)
            par_grad = par_grad.to(self.device)
            par.backward(par_grad)
            opt.step()
            y_means.append(ys.mean().item())
            costs_means.append(np.mean(costs).item())

        return emb, psi_net, y_means, [None], costs_means

    def stp_softmax_grad_par(self, par, sample_size, y_bl, instance, select='gumbel'):
        state, info = self.env.reset(instance=instance, num_instance=sample_size)
        terminated = [False]
        par_adv = (par - par.mean())

        cumulative_reward = 0
        rewards = []
        logits = []
        probs = []
        available_edges = []
        actions = []

        while not all(terminated):
            self.get_action_prob_from_state_and_heatmap(state, par_adv, select=select)
            print('hi')

    def get_action_prob_from_state_and_heatmap(self, state, par_adv, select, temperature=1.0):
        heatmap_stack = par_adv.repeat_interleave(len(state))

