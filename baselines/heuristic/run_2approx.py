import os
import re
import glob
import pickle
import pathlib
from baselines.heuristic.solver import ParallelSolver
from baselines.heuristic.solver import Solver


def extract_nsamples(s):
    match = re.search(r'_n(\d+)_', s)
    if match:
        return int(match.group(1))
    return None


if __name__ == "__main__":
    path_dir_testdata = 'synthetic_datasets/testdata_gaussian/'

    path_dir_result = 'logs/baseline@2approx/'
    os.makedirs(path_dir_result, exist_ok=True)

    for dir_name in glob.glob(f'{path_dir_testdata}/*'):
        
        # nsamples = extract_nsamples(str(dir_name))
        # if nsamples > 1001: continue
        
        # read dataset
        data = []
        for path_pkl in glob.glob(f'{dir_name}/*.pkl'):
            with open(path_pkl, 'rb') as file:
                dt = pickle.load(file)
                data.extend(dt)
        print(dir_name)
        # solve 
        # solver = Solver()
        # index, problem, solution = data[0]
        # solver.solve(index, problem, solution)
        
        solver = ParallelSolver(ncores=-1)
        results = solver.run(data)
        
        dir_name = pathlib.Path(dir_name)
        path_output = f"{path_dir_result}/{dir_name.stem}@2approx.pkl"
        with open(path_output, 'wb') as file:
            pickle.dump(results, file)

    