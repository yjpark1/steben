import multiprocessing as mp
import os
import pickle
import time

import numpy as np
import ray
import tqdm
from ray.util.queue import Queue

from stpgen.datasets.synthetic.instance import (STPInstance_erdos_renyi,
                                                 STPInstance_grid,
                                                 STPInstance_regular,
                                                 STPInstance_watts_strogatz)
from stpgen.solvers.cvxpystp import CVXSTPvec
from stpgen.solvers.heuristic import TwoApproximation
from stpgen.solvers.scipjack import SCIPJackSTP


@ray.remote
class SolverActor:
    def __init__(self, queue, parameters):
        self.queue = queue
        self.parameters = parameters
        
    def produce(self, batch_index, indexes, seeds):
        dataset = []
        for index, seed in zip(indexes, seeds):
            output = self.process(index, seed, self.parameters)
            dataset.append(output)
        self.queue.put((batch_index, dataset))
        
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
        
    def _solve_instance(self, instance, method, lock=None):
        if method == 'scipjack':
            solver = SCIPJackSTP(graph=instance, lock=lock, timelimit=100)
            
        elif method == 'cvxpystp':
            solver = CVXSTPvec(graph=instance, timelimit=100)
            
        elif method == '2approx':
            solver = TwoApproximation(graph=instance)
            
        else:
            raise NotImplementedError(f"Unsupported solving method {method}")
        solver.solve(verbose=False)
        return solver.solution

    def process(self, index, seed, parameters):
        inst_func = self._get_instance_generation_function(parameters)
        inst_func = inst_func(**parameters, seed=seed)
        instance = inst_func.sample()
        
        method = parameters['solver']
        solution = self._solve_instance(instance, method)
        
        return (index, instance, solution)
        

@ray.remote
class SaveActor:
    def __init__(self, queue, num_batches, save_dir):
        self.queue = queue
        self.num_batches = num_batches
        self.save_dir = save_dir
        
    def consume(self):
        # pbar = tqdm.tqdm(total=self.num_batches)
        while True:
            item = self.queue.get()
            if item is None:  # Signal to stop
                # pbar.close()
                break
            else:
                self._save_batch_to_pickle(item)
                # pbar.update(1)
    
    def _save_batch_to_pickle(self, batch):
        batch_index, batch = batch
        digit = np.ceil(np.log10(self.num_batches)).astype(int)
        batch_index = str(f"%0{digit}d" %(batch_index))
        filename = f"{self.save_dir}/batch_{batch_index}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(batch, file)


class RayLabeledDatasetGeneratorV3:
    def __init__(self, n: int, parameters: dict, seed: int = None, 
                 ncores: int = 1, batchsize: int = None, save_dir: str = None) -> None:
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
        self.seed = np.random.SeedSequence(seed).spawn(self.n)
        self.ncores = ncores if ncores > 0 else mp.cpu_count()
        self.queue = Queue(self.ncores)
        self.MAX_NUM_PENDING_TASKS = self.ncores * 2
        self.batchsize = batchsize
        self.num_batches = np.ceil(self.n / self.batchsize).astype(int)
        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        
        if not ray.is_initialized():
            ray.init(num_cpus=self.ncores)
            print("init done.")
        
    def run(self, filename=None):
        ns_ = np.array_split(np.arange(self.n), self.num_batches)
        seeds_ = np.array_split(self.seed, self.num_batches)

        # Start multiple producers
        producers = [SolverActor.remote(self.queue, self.parameters) for _ in range(self.ncores - 1)]
        
        # Start a single consumer
        consumer = SaveActor.remote(self.queue, self.num_batches, self.save_dir)
        producer_tasks = []
        for i in range(self.num_batches):
            producer_index = i % (self.ncores - 1)
            task = producers[producer_index].produce.remote(i, ns_[i], seeds_[i])
            producer_tasks.append(task)
        
        consumer_task = consumer.consume.remote()

        # Wait for producers to finish producing
        pbar = tqdm.tqdm(total=self.num_batches)
        while len(producer_tasks):
            done, producer_tasks = ray.wait(producer_tasks, num_returns=1)
            ray.get(done)
            pbar.update(1)
        pbar.close()

        # Signal the consumer to stop
        self.queue.put(None)

        # Wait for the consumer to finish consuming
        ray.get(consumer_task)
