import glob
import pickle
import time
from copy import deepcopy

import numpy as np
import ray
import torch
import tqdm

from stpgen.datasets.synthetic.instance import MAX_EDGE_COST
from stpgen.solvers.heuristic import TwoApproximation, remove_inessentials
from networkx.algorithms.approximation import steiner_tree


def evaluation(args, env, agent, iterations=1):
    # load the best model
    path = f"{args.log_dir}/{args.filename_for_dqn_weights}.pth"
    agent.network.load_state_dict(torch.load(path))
    agent.network.eval()
    agent.epsilon = 0.
    
    terminal_edge_ratio = []
    
    total_reward_eval = []
    cost_eval = []
    gap_eval = []
    answers = []
    for episode in tqdm.tqdm(range(200)):  # Run 100 episodes
        obs, infos = env.reset()
        # sol = steiner_tree(env.instance, env.instance.graph['Terminals']['terminals'], weight='cost')
        
        solver = TwoApproximation(env.instance)
        solver.solve()
        solver.solution = env.instance.edge_subgraph(solver.solution.edges).copy()
        answers.append(solver.solution)
        
        c_huer = np.sum(list(map(lambda x: x[2]['cost'], solver.solution.edges(data=True)))) / MAX_EDGE_COST
        inst = env.instance.copy()
        # print(inst.graph['Terminals'], 'edges: ', solver.solution.number_of_edges())
        # r = solver.solution.number_of_edges() / inst.graph['Terminals']['meta']['numTerminals']
        # terminal_edge_ratio.append(r)
        
        # sol = steiner_tree(env.instance, env.instance.graph['Terminals']['terminals'], weight='cost', method='kou')
        # c_huer = np.sum(list(map(lambda x: x[2]['cost'], sol.edges(data=True)))) / MAX_EDGE_COST
        
        total_reward_eval_repeat = []
        cost_eval_repeat = []
        for _ in range(iterations):
            state, info = env.reset(instance=inst)
            total_reward = 0
            terminated = False
            while not terminated:
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                state = next_state
                
                # logging
                total_reward += reward
                if terminated:
                    total_reward_eval_repeat.append(total_reward)
                    # cost_eval_repeat.append(env._get_current_cost())
                    subgraph, cost = get_subgraph_from_state(state)
                    cost_eval_repeat.append(cost)
                    break
        
        idx = np.argmin(cost_eval_repeat)
        total_reward_eval.append(total_reward_eval_repeat[idx])
        cost_eval.append(cost_eval_repeat[idx])
        gap = cost_eval[-1] / c_huer
        gap_eval.append(gap)
    print(f"reward: {np.mean(total_reward_eval): .3f}, cost: {np.mean(cost_eval): .3f}, gap: {np.mean(gap_eval): .6f}, gap: {np.std(gap_eval): .6f}")
    # print(np.mean(terminal_edge_ratio))
    
    data = {'reward': total_reward_eval, 'cost': cost_eval, 'gap': gap_eval, 'sol': answers}
    with open(args.log_dir + f'/evaluation.pkl', 'wb') as f:
        pickle.dump(data, f)
        
        
def inference_and_evaluation(args, env, agent, iterations=1):
    if args.testdata_dir is None:
        return 
    
    is_sampling = iterations > 1
    
    # load the best model
    path = f"{args.log_dir}/{args.filename_for_dqn_weights}.pth"
    agent.network.load_state_dict(torch.load(path))
    agent.network.eval()
    agent.epsilon = 0.
    
    # if iterations > 1:
    #     processor = PostProcessActorManager(num_workers=iterations)
    
    results = {}
    gap_eval = []
    # load test data
    durations = 0
    for path_data in glob.glob(args.testdata_dir + "/*.pkl"):
        with open(path_data, "rb") as f:
            data = pickle.load(f)    
            t0 = time.time()
            
            for ind, instance, solution in tqdm.tqdm(data):
                opt_cost = solution.graph['Info']['cost']
                # eval_repeat = {}
                
                # envs = ParallelEnv(env, iterations, max_edge_cost=1.)
                # state, info = envs.reset([instance] * iterations)
                
                # terminated = False 
                # while not terminated:
                #     action = agent.get_action(state, sampling=is_sampling)
                #     next_state, reward, terminated, truncated, info = envs.step(action)
                #     state = next_state
                #     terminated = all(terminated)
                #     # logging
                #     if terminated:
                #         state = [deepcopy(e.state) for e in envs.envs]
                #         break
                
                # for s in state:
                #     print(s.partial_solution)
                    # print(len(s.partial_solution))
                cost_eval_repeat = []
                for i in range(iterations):
                    state, info = env.reset(instance=instance)
                    env.max_edge_cost = 1
                    terminated = False
                    while not terminated:
                        action = agent.get_action(state, sampling=is_sampling)
                        next_state, reward, terminated, truncated, info = env.step(action)
                        state = next_state
                        # logging
                        if terminated:
                            # eval_repeat[i] = deepcopy(state)
                            subgraph, cost = get_subgraph_from_state(state)
                            print(state.partial_solution, cost, opt_cost)
                            cost_eval_repeat.append(cost)
                            break
                
                if iterations > 1:
                    # output = processor.run(list(eval_repeat.values()))
                    # subgraph, c = sorted(output, key=lambda x: x[1])[0]
                    idx = np.argmin(cost_eval_repeat)
                    c = cost_eval_repeat[idx]
                    state.partial_solution
                    results[ind] = c
                else:
                    # subgraph, c = get_subgraph_from_state(state)
                    results[ind] = c
                gap_eval.append(c / opt_cost)
            t1 = time.time() - t0
            durations += t1
    
    print(f"cost: {np.mean(list(results.values())): .1f}, gap: {np.mean(gap_eval): .6f}, time: {durations:.3f}")
    with open(args.log_dir + f'/evaluation_inference@{args.num_inferences}.pkl', 'wb') as f:
        pickle.dump(results, f)


def get_subgraph_from_state(state):
    graph = state.graph
    solution = state.partial_solution
    edges_sol = []
    if len(solution) > 1:
        for i, current_action in enumerate(solution[1:]):
            partial_solution_prev = solution[:(i+1)]
            edges = [((current_action, n), graph[current_action][n]) for n in partial_solution_prev if graph.has_edge(current_action, n)]
            edge_selected = sorted(edges, key=lambda x: x[1]['cost'])[0]
            edges_sol.append(edge_selected[0])
    
    subgraph = graph.edge_subgraph(edges_sol)
    terminals = graph.graph['Terminals']['terminals']
    subgraph = remove_inessentials(subgraph.copy(), terminals)
    
    cost = 0 
    for _, _, dt in subgraph.edges(data=True):
        cost += dt['cost']
    
    return subgraph, cost

class PostProcess:
    def __init__(self ):
        pass    
    
    def run(self, state):
        subgraph, cost = get_subgraph_from_state(state)
        return (subgraph, cost)

@ray.remote
class PostProcessActor(PostProcess):
    def __init__(self):
        super().__init__()
    

class PostProcessActorManager:
    def __init__(self, num_workers=1):
        ray.init(ignore_reinit_error=True)
        self.actors = [PostProcessActor.remote() for _ in range(num_workers)]
    
    def run(self, states):
        results = []
        for actor, state in zip(self.actors, states):
            results.append(actor.run.remote(state))
        return ray.get(results)
    
    def shutdown(self):
        ray.shutdown()
        
    def __del__(self):
        self.shutdown()
    
    
    
class ParallelEnv:
    def __init__(self, env, num_envs=1, max_edge_cost=1):
        self.envs = [deepcopy(env) for _ in range(num_envs)]
        self.max_edge_cost = max_edge_cost
        self.terminateds = [False] * num_envs
        self.cache_prev_nextstate = None
        
        rngs = env.rng.spawn(num_envs)
        for e, rng in zip(self.envs, rngs):
            e.rng = rng
            e.max_edge_cost = self.max_edge_cost
    
    def reset(self, instances):
        results = [env.reset(instance=inst) for env, inst in zip(self.envs, instances)]
        
        # Unzip the results
        states, infos = zip(*results)
        
        # Convert tuples to lists
        states = list(states)
        infos = list(infos)
        return states, infos
        
    def step(self, actions):
        # results = [env.step(action) if not t else self.cache_prev_nextstate[i] for i, (env, action, t) 
        #            in enumerate(zip(self.envs, actions, self.terminateds))]
        results = []
        for i, (env, action, t) in enumerate(zip(self.envs, actions, self.terminateds)):
            if t:
                result = self.cache_prev_nextstate[i]
            else:
                result = env.step(action)
            results.append(result)
        
        self.cache_prev_nextstate = results
        
        # Unzip the results
        next_states, rewards, terminateds, truncateds, infos = zip(*results)
        
        # Convert tuples to lists
        next_states = list(next_states)
        rewards = list(rewards)
        terminateds = list(terminateds)
        truncateds = list(truncateds)
        infos = list(infos)
        
        # keep terminateds
        self.terminateds = [any([t0, t1]) for t0, t1 in zip(self.terminateds, terminateds)]
        return next_states, rewards, self.terminateds, truncateds, infos
    
    def __del__(self):
        for env in self.envs:
            del env