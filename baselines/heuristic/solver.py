
import ray
import tqdm
import pathlib
import multiprocessing as mp
import networkx as nx
from stpgen.solvers.heuristic import TwoApproximation
from ray.util.actor_pool import ActorPool
from collections import defaultdict
import numpy as np




class Solver:
    def __init__(self):
        pass
        
    def solve(self, index, problem, solution=None):
        try:
            solver = self._solve_instance(problem)
        except Exception as e:
            result = (index, None)
        else:
            result = self._get_result(solver, solution)
            result = (int(index), result)
        finally:
            return result
        
    def _solve_instance(self, instance):
        solver = TwoApproximation(instance)
        solver.solve()
        solver.solution = instance.edge_subgraph(solver.solution.edges).copy()
        return solver
    
    def _get_result(self, solver, solution_opt):
        # time_scipjack, time_2approx, cost_scipjack, cost_2approx, gap
        cost_solver = self._get_cost(solver.solution)
        gap = self._get_gap(solver.solution, solution_opt)
        re = {
            'time_scipjack': solution_opt.graph['Info']['time'],
            'time_2approx': solver.duration, 
            'cost_scipjack': solution_opt.graph['Info']['cost'], 
            'cost_2approx': cost_solver, 
            'gap': gap
        }
        return re
    
    def _get_cost(self, solution):
        return np.sum([dt['cost'] for _, _, dt in solution.edges(data=True)])
        
    def _get_gap(self, solution_solver, solution_opt):
        if solution_opt is not None:
            c_solver = self._get_cost(solution_solver)
            c_opt = solution_opt.graph['Info']['cost']
            gap = (c_solver - c_opt) / c_opt
            return gap
        return None

@ray.remote
class SolverActor(Solver):
    def __init__(self):
        super().__init__()


class ParallelSolver:
    def __init__(self, ncores: int = 1) -> None:
        self.ncores = ncores if ncores > 0 else mp.cpu_count()
        self.num_solvers = self.ncores - 1
        
        if not ray.is_initialized():
            ray.init(num_cpus=self.ncores)
            print("init done.")
        
    def run(self, dataset: list):
        # Initialize actors
        solvers = [SolverActor.remote() for i in range(self.num_solvers)]

        # Initialize ActorPool with solvers
        pool_solvers = ActorPool(solvers)

        # Assign tasks to solvers
        task_to_actor_map = {}
        problem_solving = []
        num_solved = 0
        results = {}
        try:
            with tqdm.tqdm(total=len(dataset)) as pbar:
                while True:
                    if pool_solvers.has_free() and len(dataset):
                        index, instance, solution = dataset.pop()
                        solver = pool_solvers.pop_idle()
                        task = solver.solve.remote(index, instance, solution)
                        problem_solving.append(task)
                        task_to_actor_map[task] = solver
                    
                    # Process finished tasks
                    finished, problem_solving = ray.wait(problem_solving, timeout=None)
                    for task in finished:
                        try:
                            result = ray.get(task)
                            # Here you can add handling of the result if needed
                        except Exception as e:
                            print(f"Error processing task: {e}")
                        else:
                            actor = task_to_actor_map.pop(task)
                            pool_solvers.push(actor)
                            
                            index, re = result
                            results[index] = re
                            
                            num_solved += 1
                            pbar.update(1)
                    
                    if len(dataset) == 0:
                        break
                                
                    
        except KeyboardInterrupt:
            print("Interrupted by user, shutting down.")
            
        else:
            print("Problem solving completed.")
            return results

        finally:
            # Gracefully handle shutdown
            self.print_gap(results)
            ray.shutdown()
    
    def _get_index_from_filename(self, filename):
        filename = pathlib.Path(filename)
        index = filename.stem.replace('inst', '')
        return int(index)
    
    def print_gap(self, results):
        gaps = []
        for idx, dt in results.items():
            gaps.append(dt['gap'])
        print(np.mean(gaps))