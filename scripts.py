import sys
import os.path
import pandas as pd
from dq_analysis.datasets.data import Data, KNOWN_DATASETS
from dq_analysis.attributes.consistency import get_consistent_dataset

def save_validated_vulnerabilities(args):
    """
    Collect vulnerabilities that have been manually validated and referenced in sample.csv
    """
    print('***********************************************')
    dataset = args[2]
    filepath = args[3]
    samplepath = f'dq_analysis/datasets/{dataset}/sample.csv'

    if not os.path.isfile(samplepath):
        print(f'no file found at {samplepath} for {dataset} dataset')
        return None
    
    print(f'Collecting manually validated vulnerabilities from {dataset} and adding them to {filepath}')
    data = Data(dataset).get_dataset()

    # Load manually inspected samples
    inspected = pd.read_csv(samplepath)
    inspected = inspected.merge(data, how='left', on=['ID', 'UID', 'Vulnerable'])

    # Save only correctly labeled vulnerable samples
    correct = inspected.loc[inspected['Label'] == 1]
    correct.to_csv(filepath, mode='a', index=False, header=False)
    print(f'{len(correct)} vulnerable samples written to {filepath} from {dataset}')

def collect_all_validated_vulnerabilties(args):
    filepath = args[2]
    for dataset in KNOWN_DATASETS:
        save_validated_vulnerabilities([0,0,dataset, filepath])
    file = pd.read_csv(filepath)
    print(f'Saved {len(file)} vulnerable samples to {filepath}')

if __name__ == '__main__':
    script = sys.argv[1]
    execute = {
        'collect_all_validated_vulnerabilties': collect_all_validated_vulnerabilties,
        'save_validated_vulnerabilities': save_validated_vulnerabilities,
    }
    execute[script](sys.argv)