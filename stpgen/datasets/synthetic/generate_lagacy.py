import multiprocessing as mp
import os
import pickle
import time

import numpy as np
import ray
import tqdm
from ray.util import queue
from ray.util.queue import Queue

from stpgen.datasets.synthetic.instance import (STPInstance_erdos_renyi,
                                                 STPInstance_grid,
                                                 STPInstance_regular,
                                                 STPInstance_watts_strogatz)
from stpgen.solvers.cvxpystp import CVXSTPvec
from stpgen.solvers.heuristic import TwoApproximation
from stpgen.solvers.scipjack import SCIPJackSTP


class Generator:
    """Generator STP instances with multiprocessing
    https://stackoverflow.com/questions/28664720/how-to-create-global-lock-semaphore-with-multiprocessing-pool-in-python
    """
    lock = mp.Lock()
    
    def __init__(self, n: int, parameters: dict, seed: int = None, ncores:int=1) -> None:
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
        self.seed = seed 
        self.seed = np.random.SeedSequence(seed).spawn(n)
        self.ncores = ncores if ncores > 0 else mp.cpu_count()
        
    def run(self, filename=None):
        pool = mp.Pool(self.ncores)
        jobs = self._make_jobs(pool)
        
        dataset = []
        for job in jobs:
            instance = job.get()
            dataset.append(instance)
            
        if filename:
            with open(filename, 'wb') as file:
                pickle.dump(dataset, file)
        
        pool.close()
        pool.join()
        
        return dataset
    
    def _make_jobs(self, pool):
        jobs = []
        for i in range(self.n):
            seed = self.seed[i]
            jobs.append(pool.apply_async(self.func, (seed, )))
        
        return jobs
    
    @classmethod
    def func(cls, seed):
        with cls.lock:
            rng = np.random.default_rng(seed)
            return rng.integers(low=0, high=10, size=3)
        

class LabeledDatasetGenerator(Generator):
    def __init__(self, n: int, parameters: dict, seed: int = None, ncores: int = 1) -> None:
        super().__init__(n, parameters, seed, ncores)
    
    def run(self, filename=None):
        pool = mp.Pool(self.ncores)
        jobs = self._make_jobs(pool)
        
        dataset = []
        for job in jobs:
            instance = job.get()
            dataset.append(instance)
            
        if filename:
            with open(filename, 'wb') as file:
                pickle.dump(dataset, file)
        
        pool.close()
        pool.join()
        
        return dataset
    
    def _make_jobs(self, pool):
        jobs = []
        for i in range(self.n):
            seed = self.seed[i]
            jobs.append(pool.apply_async(self.func, (i, seed, self.parameters)))
        return jobs
    
    @staticmethod
    def _get_instance_generation_function(parameters):
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
    
    @staticmethod
    def _solve_instance(instance, method, lock):
        if method == 'scipjack':
            solver = SCIPJackSTP(graph=instance, lock=lock, timelimit=20)
            
        elif method == 'cvxpystp':
            solver = CVXSTPvec(graph=instance, timelimit=20)
            
        else:
            raise NotImplementedError(f"Unsupported solving method {method}")
        solver.solve(verbose=False)
        return solver.solution
    
    def func(self, index, seed, parameters):
        inst_func = self._get_instance_generation_function(parameters)
        inst_func = inst_func(**parameters, seed=seed)
        instance = inst_func.sample()
        
        method = 'cvxpystp'
        solution = self._solve_instance(instance, method, lock=self.lock)
        
        return (instance, solution.edges())


def _get_instance_generation_function(parameters):
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

    
def _solve_instance(instance, method, lock=None):
    if method == 'scipjack':
        solver = SCIPJackSTP(graph=instance, lock=lock, timelimit=100)
        
    elif method == 'cvxpystp':
        solver = CVXSTPvec(graph=instance, timelimit=1e+4)
        
    elif method == '2approx':
        solver = TwoApproximation(graph=instance)
        
    else:
        raise NotImplementedError(f"Unsupported solving method {method}")
    solver.solve(verbose=False)
    return solver.solution


def process(index, seed, parameters):
    inst_func = _get_instance_generation_function(parameters)
    inst_func = inst_func(**parameters, seed=seed)
    instance = inst_func.sample()
    
    method = parameters['solver']
    solution = _solve_instance(instance, method)
    
    return (index, instance, solution)

@ray.remote
def worker(i, indexes, seeds, parameters):
    dataset = []
    for index, seed in zip(indexes, seeds):
        output = process(index, seed, parameters)
        dataset.append(output)
    return i, dataset


class RayLabeledDatasetGeneratorV2(Generator):
    def __init__(self, n: int, parameters: dict, seed: int = None, ncores: int = 1, batchsize: int = None,
                 save_dir: str = None) -> None:
        super().__init__(n, parameters, seed, ncores)
        self.is_run_async = True
        self.batchsize = batchsize
        self.num_batches = np.ceil(self.n / self.batchsize).astype(int)
        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        
        if not ray.is_initialized():
            ray.init(num_cpus=self.ncores)
            
    def __del__(self):
        ray.shutdown()
    
    def run(self, filename=None):
        ns_ = np.array_split(np.arange(self.n), self.num_batches)
        seeds_ = np.array_split(self.seed, self.num_batches)
        
        jobs = []
        for i in range(self.num_batches):
            # worker(ns_[i], seeds_[i], self.parameters)
            jobs.append(worker.remote(i, ns_[i], seeds_[i], self.parameters))
            
        t0 = time.time()
        if self.is_run_async:
            dataset = self._run_wait(jobs)
        else:
            dataset = self._run_get(jobs)
        print(f"Duration: {time.time() - t0:.3f} sec.")
        
        if filename:
            with open(filename, 'wb') as file:
                pickle.dump(dataset, file)
        return dataset
    
    def _run_get(self, jobs):
        dataset = []
        for gen in jobs:
            i, o = ray.get(gen)
            dataset += o
        # dataset = sorted(dataset, key=lambda x: x[0])
        return dataset
    
    def _run_wait(self, jobs):
        pbar = tqdm.tqdm(total=self.num_batches)
        while len(jobs):
            done, jobs = ray.wait(jobs)
            o = ray.get(done[0])
            self._save_batch_to_pickle(o)
            pbar.update(1)
        pbar.close()
        return None
    
    def _save_batch_to_pickle(self, batch):
        batch_index, batch = batch
        digit = np.ceil(np.log10(self.num_batches)).astype(int)
        # indexes = [x[0] for x in batch]
        # batch_index = np.ceil(min(indexes) / self.num_batches).astype(int)
        batch_index = str(f"%0{digit}d" %(batch_index))
        filename = f"{self.save_dir}/batch_{batch_index}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(batch, file)


@ray.remote
class Producer:
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
            solver = CVXSTPvec(graph=instance, timelimit=1e+4)
            
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
class Consumer:
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


class RayLabeledDatasetGeneratorV3(RayLabeledDatasetGeneratorV2):
    def __init__(self, n: int, parameters: dict, seed: int = None, ncores: int = 1, batchsize: int = None, save_dir: str = None) -> None:
        super().__init__(n, parameters, seed, ncores, batchsize, save_dir)
        self.queue = Queue(self.ncores)
        self.MAX_NUM_PENDING_TASKS = self.ncores * 2
    
    def _run(self, filename=None):
        ns_ = np.array_split(np.arange(self.n), self.num_batches)
        seeds_ = np.array_split(self.seed, self.num_batches)
        
        pbar = tqdm.tqdm(total=self.num_batches)
        result_refs = []
        for i in range(self.num_batches):
            if len(result_refs) > self.MAX_NUM_PENDING_TASKS:
                # update result_refs to only track the remaining tasks.
                ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
                batch = ray.get(ready_refs)
            result_refs.append(worker.remote(i, ns_[i], seeds_[i], self.parameters))
            pbar.update(1)
        out = ray.get(result_refs)
        pbar.close()
        
    def run(self, filename=None):
        ns_ = np.array_split(np.arange(self.n), self.num_batches)
        seeds_ = np.array_split(self.seed, self.num_batches)

        # Start multiple producers
        producers = [Producer.remote(self.queue, self.parameters) for _ in range(self.ncores - 1)]
        
        # Start a single consumer
        consumer = Consumer.remote(self.queue, self.num_batches, self.save_dir)
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
        
    def _run(self, filename=None):
        """failed.
        unsynchronized process beween producer & consumer
        """
        ns_ = np.array_split(np.arange(self.n), self.num_batches)
        seeds_ = np.array_split(self.seed, self.num_batches)

        # Start multiple producers
        producers = [Producer.remote(self.queue, self.parameters) for _ in range(self.ncores - 1)]
        
        # Start a single consumer
        consumer = Consumer.remote(self.queue, self.num_batches, self.save_dir)

        # Get handles to producer and consumer tasks
        pbar = tqdm.tqdm(total=self.num_batches)
        producer_tasks = []
        for i in range(self.num_batches):
            producer_index = i % (self.ncores - 1)
            task = producers[producer_index].produce.remote(i, ns_[i], seeds_[i])
            producer_tasks.append(task)
            if len(producer_tasks) > self.MAX_NUM_PENDING_TASKS:
                # update result_refs to only track the remaining tasks.
                done, producer_tasks = ray.wait(producer_tasks, num_returns=1)
                ray.get(done)
            pbar.update(1)
        pbar.close()
        
        consumer_task = consumer.consume.remote()
        
        # Signal the consumer to stop
        self.queue.put(None)

        # Wait for the consumer to finish consuming
        ray.get(consumer_task)
