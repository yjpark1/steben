import gc
import glob
import multiprocessing as mp
import os
import pathlib
import pickle
import shutil
import time
from collections import defaultdict

import numpy as np
import ray
import tqdm
from ray._private.utils import get_system_memory, get_used_memory
from ray.util.actor_pool import ActorPool
from ray.util.queue import Queue

from stpgen.datasets.io import SCIPParser, SteinLibFormatWriter
from stpgen.datasets.synthetic.instance import (STPInstance_erdos_renyi,
                                                 STPInstance_grid,
                                                 STPInstance_regular,
                                                 STPInstance_watts_strogatz)
# from stpgen.solvers.cvxpystp import CVXSTPvec
# from stpgen.solvers.heuristic import TwoApproximation
from stpgen.solvers.scipjack import SCIPJackSTP


@ray.remote
class ProblemWriterActor:
    def __init__(self, path_dir, parameters, seed):
        self.path_dir = path_dir
        self.parameters = parameters
        self.seed = seed
        os.makedirs(f"{self.path_dir}/stpfile/", exist_ok=True)
        self.rng = np.random.default_rng(self.seed)
            
    def write(self):
        index = 0
        while True:
            numfiles = len(glob.glob(f"{self.path_dir}/stpfile/inst*.stp"))
            if numfiles < 500:
                index, inst = self._generate_instance(index, self.parameters)
                writer = SteinLibFormatWriter(inst)
                filepath = f"{self.path_dir}/stpfile/inst{index}.stp"
                writer.write(filepath)
                index += 1
            else:
                time.sleep(1)
            
    def _generate_instance(self, index, parameters):
        inst_func = self._get_instance_generation_function(parameters)
        inst_func = inst_func(**parameters, rng=self.rng)
        instance = inst_func.sample()
        return (index, instance)
            
    def _get_instance_generation_function(self, parameters):
        if parameters['graph_type'] == 'erdos_renyi':
            inst_func = STPInstance_erdos_renyi
        elif parameters['graph_type'] == 'watts_strogatz':
            inst_func = STPInstance_watts_strogatz
        elif parameters['graph_type'] == 'regular':
            inst_func = STPInstance_regular
        elif parameters['graph_type'] == 'grid':
            inst_func = STPInstance_grid
        else:
            raise NotImplementedError('Unsupported graph type for generation.')
        return inst_func
    
    def stop(self):
        ray.actor.exit_actor()
    

@ray.remote
class SolverActor:
    def __init__(self, path_dir, parameters, queue=None):
        self.path_dir = path_dir
        self.parameters = parameters
        self.queue = queue
        
    def solve(self, path_file, include_solution=True):
        try:
            path_file = pathlib.Path(path_file)
            index = path_file.stem.replace('inst', '')
            problem, solution = self._solve_instance(path_file, include_solution)
            
        except Exception as e:
            result = (index, None, None)
        else:
            result = (int(index), problem, solution)
        finally:
            os.remove(path_file)
            return result
        
    def _solve_instance(self, instance, include_solution):
        method = self.parameters['solver']
        if method == 'scipjack':
            solver = SCIPJackSTP(path=instance, include_solution=include_solution)
        else:
            raise NotImplementedError(f"Unsupported solving method {method}")
        solver.solve(verbose=False)
        return solver.graph, solver.solution
        

# @ray.remote
# class SaveActor:
#     def __init__(self, save_dir, num_samples):
#         self.save_dir = save_dir
#         self.num_samples = num_samples
#         self.result = []
        
#     def save(self, solution_queue):
#         while True:
#             if not solution_queue.empty():
#                 item = solution_queue.get(timeout=None)
#                 if item:
#                     self.result.append(item)
#                     # self._save_item_to_pickle(item)
#             else:
#                 time.sleep(0.1)
                
#             if len(self.result) == self.n:
#                 index = 'full'
#                 filename = f"{self.save_dir}/instance_{index}.pkl"
#                 with open(filename, 'wb') as file:
#                     pickle.dump(item, file)
#                 break
                
#     def _save_item_to_pickle(self, item):
#         # index, problem, solution = item
#         # filename = f"{self.save_dir}/instance_{index}.pkl"
#         # with open(filename, 'wb') as file:
#         #     pickle.dump(item, file)
#         index = np.random.randint(0, high=10)
#         filename = f"{self.save_dir}/instance_{index}.pkl"
#         with open(filename, 'wb') as file:
#             pickle.dump(item, file)
            
#     def stop(self):
#         ray.actor.exit_actor()
        

@ray.remote
class SaveActor:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        
    def save(self, solution_queue, index):
        self._save_item_to_pickle(solution_queue, index)
                
    def _save_item_to_pickle(self, results, index):
        filename = f"{self.save_dir}/batch_{index}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(results, file)
            
    def stop(self):
        ray.actor.exit_actor()
    

class RayLabeledDatasetGeneratorV4:
    def __init__(self, n: int, parameters: dict, seed: int = None, 
                 ncores: int = 1, batchsize: int = None, save_dir: str = None, 
                 include_solution: bool = True) -> None:
        """
        Args:
            n (int): number of instance to generate
            parameters (dict): parameter of instance distribution to control graph structure 
            seed (int, optional): entropy in SeedSequence. see https://numpy.org/doc/stable/reference/random/bit_generators/index.html#seeding-and-entropy. Defaults to None.
            ncores (int, optional): cpu cores to multiprocess. Defaults to 1.
            lock (_type_, optional): global lock for file I/O. Defaults to None.
        """
        self.n = n
        self.parameters = parameters
        self.ncores = ncores if ncores > 0 else mp.cpu_count()
        # self.queue = Queue(self.ncores * 2)
        self.num_solvers = self.ncores - 3
        self.seed = np.random.SeedSequence(seed).spawn(1)
        self.queue = None # Queue(maxsize=self.num_solvers * 2, actor_options={'num_cpus': 1})
        self.save_dir = save_dir
        self.include_solution = include_solution
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        
        if not ray.is_initialized():
            ray.init(num_cpus=self.ncores)
            print("init done.")
        
    def run(self, filename=None):
        # Initialize actors
        writers = ProblemWriterActor.remote(self.save_dir, self.parameters, self.seed[0])
        solvers = [SolverActor.remote(self.save_dir, self.parameters, self.queue) for i in range(self.num_solvers)]
        # saver = SaveActor.remote(self.save_dir, self.n)

        # Start writing problems asynchronously
        problem_writing_future = writers.write.remote()

        # Start the saving task asynchronously
        # solution_saving_future = saver.save.remote(self.queue)

        # Initialize ActorPool with solvers
        pool_solvers = ActorPool(solvers)

        # Monitor directory for new files and assign tasks to solvers
        pattern = f"{self.save_dir}/stpfile/inst*.stp"
        task_to_actor_map = {}
        problem_solving = []
        assigned_file_index = defaultdict(int)
        num_solved = 0
        results = []
        try:
            with tqdm.tqdm(total=self.n) as pbar:
                while True:
                    files = glob.glob(pattern)
                    files = [f for f in files if assigned_file_index[self._get_index_from_filename(f)] == 0]
                    
                    for path_file in files:
                        if pool_solvers.has_free():
                            solver = pool_solvers.pop_idle()
                            task = solver.solve.remote(path_file, include_solution=self.include_solution)
                            problem_solving.append(task)
                            task_to_actor_map[task] = solver
                            assigned_file_index[self._get_index_from_filename(path_file)] = 1
                    
                    # Process finished tasks
                    finished, problem_solving = ray.wait(problem_solving, timeout=None)
                    for task in finished:
                        try:
                            result = ray.get(task)
                            # Here you can add handling of the result if needed
                        except Exception as e:
                            print(f"Error processing task: {e}")
                        else:
                            index, prob, sol = result
                            actor = task_to_actor_map.pop(task)
                            pool_solvers.push(actor)
                            
                            if self.include_solution:
                                if sol:
                                    result = (num_solved, prob, sol)
                                    results.append(result)
                                    
                                    num_solved += 1
                                    pbar.update(1)
                            else:
                                result = (num_solved, prob, sol)
                                results.append(result)
                                
                                num_solved += 1
                                pbar.update(1)
                    
                    if len(results) == self.n:
                        writers.stop.remote()
                        break
                    
                    # if num_solved == self.n:
                    #     tmpdir = f"{self.save_dir}/stpfile/"
                    #     if len(os.listdir()) == 0:
                    #         shutil.rmtree(tmpdir, ignore_errors=True)
                    #         saver.stop.remote()
                    #         break
                    
                    # if num_solved == self.n:
                    #     if len(glob.glob(f"{self.save_dir}/instance_*.pkl")) == self.n:
                    #         saver.stop.remote()
                    #         break
                        
        except KeyboardInterrupt:
            print("Interrupted by user, shutting down.")
            
        else:
            self._save_results(results)
            print("Dataset generation completed.")

        finally:
            # Gracefully handle shutdown
            ray.shutdown()
            tmpdir = f"{self.save_dir}/stpfile"
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    def _get_index_from_filename(self, filename):
        filename = pathlib.Path(filename)
        index = filename.stem.replace('inst', '')
        return int(index)
    
    def _save_results(self, results):
        results = sorted(results, key=lambda x: x[0])
        index = 'full'
        filename = f"{self.save_dir}/instance_{index}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(results, file)
        
        
class RayLabeledDatasetGeneratorV5:
    """Large dataset generator"""
    def __init__(self, n: int, parameters: dict, seed: int = None, 
                 ncores: int = 1, batchsize: int = 10000, save_dir: str = None,
                 include_solution: bool = True) -> None:
        """
        Args:
            n (int): number of instance to generate
            parameters (dict): parameter of instance distribution to control graph structure 
            seed (int, optional): entropy in SeedSequence. see https://numpy.org/doc/stable/reference/random/bit_generators/index.html#seeding-and-entropy. Defaults to None.
            ncores (int, optional): cpu cores to multiprocess. Defaults to 1.
            lock (_type_, optional): global lock for file I/O. Defaults to None.
        """
        self.n = n
        self.parameters = parameters
        self.ncores = ncores if ncores > 0 else mp.cpu_count()
        self.batchsize = batchsize
        self.num_solvers = self.ncores - 3
        self.seed = np.random.SeedSequence(seed).spawn(1)
        self.save_dir = save_dir
        self.include_solution = include_solution
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        
        if not ray.is_initialized():
            ray.init(num_cpus=self.ncores)
            print("init done.")
            
    def run(self, filename=None):
        # Initialize actors
        writers = ProblemWriterActor.remote(self.save_dir, self.parameters, self.seed[0])
        solvers = [SolverActor.remote(self.save_dir, self.parameters) for i in range(self.num_solvers)]
        saver = SaveActor.remote(self.save_dir)

        # Start writing problems asynchronously
        problem_writing_future = writers.write.remote()

        # Initialize ActorPool with solvers
        pool_solvers = ActorPool(solvers)

        # Monitor directory for new files and assign tasks to solvers
        pattern = f"{self.save_dir}/stpfile/inst*.stp"
        task_to_actor_map = {}
        problem_solving = []
        assigned_file_index = defaultdict(int)
        num_solved = 0
        num_batch = 0
        results = []
        try:
            with tqdm.tqdm(total=self.n) as pbar:
                while True:
                    files = glob.glob(pattern)
                    files = [f for f in files if assigned_file_index[self._get_index_from_filename(f)] == 0]
                    
                    for path_file in files:
                        if pool_solvers.has_free():
                            solver = pool_solvers.pop_idle()
                            task = solver.solve.remote(path_file, include_solution=self.include_solution)
                            problem_solving.append(task)
                            task_to_actor_map[task] = solver
                            assigned_file_index[self._get_index_from_filename(path_file)] = 1
                            
                    # Process finished tasks
                    finished, problem_solving = ray.wait(problem_solving, timeout=None)
                    for task in finished:
                        try:
                            result = ray.get(task)
                            # Here you can add handling of the result if needed
                        except Exception as e:
                            print(f"Error processing task: {e}")
                        else:
                            index, prob, sol = result
                            actor = task_to_actor_map.pop(task)
                            pool_solvers.push(actor)
                            
                            if self.include_solution:
                                if sol:
                                    result = (num_solved, prob, sol)
                                    results.append(result)
                                    
                                    num_solved += 1
                                    pbar.update(1)
                            else:
                                result = (num_solved, prob, sol)
                                results.append(result)
                                
                                num_solved += 1
                                pbar.update(1)
                    
                    if len(results) == self.batchsize:
                        saver.save.remote(results, num_batch)
                        results = []
                        num_batch += 1
                    
                    if num_solved == self.n:
                        writers.stop.remote()
                        break
                        
        except KeyboardInterrupt:
            print("Interrupted by user, shutting down.")
            
        else:
            task = saver.save.remote(results, num_batch)
            ray.get(task)
            print("Dataset generation completed.")

        finally:
            # Gracefully handle shutdown
            ray.shutdown()
            tmpdir = f"{self.save_dir}/stpfile"
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    def _get_index_from_filename(self, filename):
        filename = pathlib.Path(filename)
        index = filename.stem.replace('inst', '')
        return int(index)
    
    # def _save_results(self, results, index):
    #     filename = f"{self.save_dir}/batch_{index}.pkl"
    #     with open(filename, 'wb') as file:
    #         pickle.dump(results, file)
        