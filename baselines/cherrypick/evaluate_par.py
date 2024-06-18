import glob
import pickle
import re
import time
from copy import deepcopy

import numpy as np
import ray
import torch
import tqdm

from stpgen.datasets.synthetic.instance import MAX_EDGE_COST
from stpgen.solvers.heuristic import TwoApproximation, remove_inessentials


def sort_key(filename):
    match = re.search(r'batch_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

        
def inference_and_evaluation(args, env, agent, iterations=1):
    if args.testdata_dir is None:
        return 
    
    is_sampling = iterations > 1
    
    # load the best model
    path = f"{args.log_dir}/{args.filename_for_dqn_weights}.pth"
    agent.network.load_state_dict(torch.load(path))
    agent.network.eval()
    agent.epsilon = 0.
    
    results = {}
    gap_eval = []
    
    envs = ParallelEnv(env, iterations, max_edge_cost=1.)
    
    durations = 0
    # load test data
    num_instance = 0
    for path_data in sorted(glob.glob(args.testdata_dir + "/*.pkl"), key=sort_key):
        with open(path_data, "rb") as f:
            data = pickle.load(f)    
            t0 = time.time()
            for ind, instance, solution in tqdm.tqdm(data):
                opt_cost = solution.graph['Info']['cost']
                eval_repeat = {}
                state, info = envs.reset([instance] * iterations)
                
                terminated = False 
                while not terminated:
                    action = agent.get_action(state, sampling=is_sampling)
                    if iterations == 1:
                        action = [action.item()]
                    next_state, reward, terminated, truncated, info = envs.step(action)
                    state = next_state
                    terminated = all(terminated)
                    # logging
                    if terminated:
                        break
                
                result_repeat = [get_subgraph_from_state(s) for s in state]
                idx = np.argmin([r[1] for r in result_repeat])
                subgraph, c = result_repeat[idx]
                results[ind] = c
                gap_eval.append(c/opt_cost)
                
                num_instance += 1
                if num_instance >= args.use_nsamples:
                    break
            
            t1 = time.time() - t0
            durations += t1
        
        if num_instance >= args.use_nsamples:
            break
                
    
    print(f"cost: {np.mean(list(results.values())): .1f}, gap: {np.mean(gap_eval): .6f}, , std: {np.std(gap_eval): .6f},  time: {durations:.3f}")
    with open(args.log_dir + f'/evaluation_inference@{args.path_inference_save}@{args.num_inferences}.pkl', 'wb') as f:
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
    
@ray.remote
class RayActorforEnv:
    def __init__(self, env):
        self.env = env
    
    def reset(self, instance):
        return self.env.reset(instance=instance)
    
    def step(self, action):
        if self.env._is_done():
            return self.env.state, 0, True, False, None
        return self.env.step(action)
    
class ParallelEnv:
    def __init__(self, env, num_envs=1, max_edge_cost=1.):
        ray.init(ignore_reinit_error=True)
        self.max_edge_cost = max_edge_cost
        self.terminateds = [False] * num_envs
        rngs = env.rng.spawn(num_envs)
        self.env_actors = []
        for i in range(num_envs):
            env.rng = rngs[i]
            env.max_edge_cost = self.max_edge_cost
            actor_env = RayActorforEnv.remote(deepcopy(env))
            self.env_actors.append(actor_env)
    
    def reset(self, instances):
        futures = []
        for actor, instance in zip(self.env_actors, instances):
            futures.append(actor.reset.remote(instance))
        
        states = []
        infos = []
        for i, future in enumerate(futures):
            state, info = ray.get(future)
            states.append(state)
            infos.append(info)
        
        return states, infos
    
    def step(self, actions):
        futures = []
        for i, (actor, action) in enumerate(zip(self.env_actors, actions)):
            futures.append(actor.step.remote(action))

        results = []
        for i, future in enumerate(futures):
            result = ray.get(future)
            results.append(result)

        # Unzip the results
        next_states, rewards, terminateds, truncateds, infos = zip(*results)

        # Convert tuples to lists
        next_states = list(next_states)
        rewards = list(rewards)
        terminateds = list(terminateds)
        truncateds = list(truncateds)
        infos = list(infos)

        return next_states, rewards, terminateds, truncateds, infos
    
    
class MutipleEnvs:
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
            
            
def inference_and_evaluation_multiple_instance(args, env, agent, batchsize=50):
    if args.testdata_dir is None:
        return 
    
    # load the best model
    path = f"{args.log_dir}/{args.filename_for_dqn_weights}.pth"
    agent.network.load_state_dict(torch.load(path))
    agent.network.eval()
    agent.epsilon = 0.
    
    results = []
    gap_eval = []
    
    envs = ParallelEnv(env, batchsize, max_edge_cost=1.)
    
    durations = 0
    # load test data
    num_instance = 0
    
    with tqdm.tqdm(total=args.use_nsamples) as pbar:
        for path_data in sorted(glob.glob(args.testdata_dir + "/*.pkl"), key=sort_key):
            with open(path_data, "rb") as f:
                data = pickle.load(f)    
                t0 = time.time()
                
                opt_costs = []
                instances = []
                for bid, (ind, instance, solution) in enumerate(data):
                    opt_cost = solution.graph['Info']['cost']
                    opt_costs.append(opt_cost)
                    instances.append(instance)
                    if bid % batchsize == 0 and bid > 0:
                        state, info = envs.reset(instances)
                        terminated = False 
                        while not terminated:
                            action = agent.get_action(state, sampling=False)
                            next_state, reward, terminated, truncated, info = envs.step(action)
                            state = next_state
                            terminated = all(terminated)
                            # logging
                            if terminated:
                                break
                        result_repeat = [get_subgraph_from_state(s) for s in state]
                        cs = [c for _, c in result_repeat]
                        results.extend(cs)
                        gap_eval.extend([c/co for c, co in zip(cs, opt_costs)])

                        opt_costs = []
                        instances = []
                        
                    num_instance += 1
                    pbar.update(1)
                    if num_instance >= args.use_nsamples:
                        break
                t1 = time.time() - t0
                durations += t1
            
            if num_instance >= args.use_nsamples:
                break
                
    
    print(f"cost: {np.mean(results): .1f}, gap: {np.mean(gap_eval): .6f}, , std: {np.std(gap_eval): .6f},  time: {durations:.3f}")
    with open(args.log_dir + f'/evaluation_inference@{args.path_inference_save}@{args.num_inferences}.pkl', 'wb') as f:
        pickle.dump(results, f)
        
        
        

def inference_and_evaluation_steinlib(args, env, agent, iterations=1):
    if args.testdata_dir is None:
        return 
    
    is_sampling = iterations > 1
    
    # load the best model
    path = f"{args.log_dir}/{args.filename_for_dqn_weights}.pth"
    agent.network.load_state_dict(torch.load(path))
    agent.network.eval()
    agent.epsilon = 0.
    
    results = {}
    gap_eval = []
    
    # envs = ParallelEnv(env, iterations, max_edge_cost=1.)
    durations = 0
    # load test data
    num_instance = 0
    path_data = args.testdata_dir
    
    with open(path_data, "rb") as f:
        data = pickle.load(f)    
        t0 = time.time()
        for ind, instance, solution in tqdm.tqdm(data):
            instance.graph['Info'].update({'max_edge_cost': np.max([d['cost'] for _, _, d in instance.edges(data=True)])})
            opt_cost = instance.graph['Info']['cost']
            eval_repeat = {}
            state, info = env.reset(instance)
            
            terminated = False 
            while not terminated:
                action = agent.get_action(state, sampling=is_sampling)
                next_state, reward, terminated, truncated, info = env.step(action)
                state = next_state
                # logging
                if terminated:
                    break
            
            subgraph, c = get_subgraph_from_state(state)
            results[ind] = c
            gap_eval.append(c/opt_cost)
            
            num_instance += 1
        t1 = time.time() - t0
        durations += t1
    
    print(f"cost: {np.mean(list(results.values())): .1f}, gap: {np.mean(gap_eval): .6f}, , std: {np.std(gap_eval): .6f},  time: {durations:.3f}")
    with open(args.log_dir + f'/evaluation_inference@{args.path_inference_save}@{args.num_inferences}.pkl', 'wb') as f:
        pickle.dump(results, f)