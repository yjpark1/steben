import sys
import os
from itertools import starmap
import shutil


sys.path.append(os.getcwd())
from stpgen.datasets.synthetic.generate_v2 import RayLabeledDatasetGeneratorV4
from stpgen.datasets.synthetic.generate_v2 import RayLabeledDatasetGeneratorV5


num_instances = 1280000
cost_type = 'uniform'
if __name__ == "__main__":    
    for method in ['scipjack']: #  '2approx'
            for num_nodes in [10, 20, 30, 50, 100]:
                
                params = [
                    {'graph_type': 'erdos_renyi', 'n': num_nodes, 'solver': method, 'cost_type': cost_type},
                    {'graph_type': 'grid', 'n': num_nodes, 'solver': method, 'cost_type': cost_type},
                    {'graph_type': 'watts_strogatz', 'n': num_nodes, 'solver': method, 'cost_type': cost_type},
                    {'graph_type': 'regular', 'n': num_nodes, 'solver': method, 'cost_type': cost_type},
                ]

                for param in params:
                    if param['graph_type'] == 'grid' and param['n'] == 10: continue
                    print(param)
                    path_to_dir = f"/root/datasets/traindata_{param['cost_type']}/synthetic_STP_dataset_train_{param['graph_type']}_n{param['n']}_{param['solver']}/"
                    shutil.rmtree(path_to_dir, ignore_errors=True)
                    # if os.path.exists(path_to_dir):
                    #     print(f"Skipping {path_to_dir}")
                    #     continue
                    generator = RayLabeledDatasetGeneratorV5(n=num_instances, parameters=param, 
                                                            seed=1234, ncores=72, save_dir=path_to_dir, 
                                                            batchsize=10000)
                    generator.run(filename=None)

