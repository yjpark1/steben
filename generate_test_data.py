import sys
import os
from itertools import starmap
import shutil


sys.path.append(os.getcwd())
from stpgen.datasets.synthetic.generate_v2 import RayLabeledDatasetGeneratorV4
from stpgen.datasets.synthetic.generate_v2 import RayLabeledDatasetGeneratorV5


cost_type = 'uniform'
if __name__ == "__main__":
    # you can set stp/reduction = 2 in stpsolver/settingsfile.set
    for method in ['scipjack']: 
        for num_nodes in [10, 20, 30, 50, 100, 200, 500, 1000]: # 
            if num_nodes < 200:
                num_instances = 10000
            elif num_nodes > 9999:
                num_instances = 30
            else:
                num_instances = 1000
            
            params = [
                {'graph_type': 'erdos_renyi', 'n': num_nodes, 'solver': method, 'cost_type': cost_type},
                {'graph_type': 'grid', 'n': num_nodes, 'solver': method, 'cost_type': cost_type},
                {'graph_type': 'watts_strogatz', 'n': num_nodes, 'solver': method, 'cost_type': cost_type},
                {'graph_type': 'regular', 'n': num_nodes, 'solver': method, 'cost_type': cost_type},
            ]

            for param in params:
                print(param)
                if param['graph_type'] == 'grid' and param['n'] == 10: continue
                path_to_dir = f"/root/datasets/testdata_{param['cost_type']}/synthetic_STP_dataset_test_{param['graph_type']}_n{param['n']}_{param['solver']}/"
                # shutil.rmtree(path_to_dir, ignore_errors=True)
                if os.path.exists(path_to_dir):
                    print(f"Skipping {path_to_dir}")
                    continue
                if num_nodes > 999:
                    generator = RayLabeledDatasetGeneratorV5(n=num_instances, parameters=param, seed=12345, ncores=72, save_dir=path_to_dir,
                                                            batchsize=100, include_solution=False)
                    generator.run(filename=None)
                else:
                    generator = RayLabeledDatasetGeneratorV4(n=num_instances, parameters=param, seed=12345, ncores=-1, save_dir=path_to_dir,
                                                            include_solution=False)
                    generator.run(filename=None)

