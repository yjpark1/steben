import glob

__dataset_names = ['SteinLib', 'Vienna', 'Copenhagen14', 'PUCN', 'GAPS']
__datasets = {k: [] for k in __dataset_names}
files = glob.iglob('data/**/raws/*')

def get_list_of_datasets(__datasets, files):
    for filepath in files:
        filepath_ = filepath.split('/')
        _, dataset_name, _, filename = filepath_
        if dataset_name in __dataset_names:
            __datasets[dataset_name].append(filepath)
        else:
            raise NotImplementedError(f"Unsupported datasets: {dataset_name}")
    return __datasets
        
__datasets = get_list_of_datasets(__datasets, files)