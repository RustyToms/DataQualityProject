import sys
import pandas as pd
from dq_analysis.datasets.data import Data
from dq_analysis.attributes.consistency import get_consistent_dataset

def save_validated_vulnerabilities(args):
    """
    Collect vulnerabilities that have been manually validated and referenced in sample.csv
    """
    print('***********************************************')
    dataset = args[2]
    filepath = args[3]
    print(f'Collecting manually validated vulnerabilities from {dataset} and adding them to {filepath}')
    if dataset != 'D2A':
        data = Data(dataset).get_dataset()
    else:
        data = get_consistent_dataset('D2A')
    # Load manually inspected samples
    inspected = pd.read_csv(f'dq_analysis/datasets/{dataset}/sample.csv')
    inspected = inspected.merge(data, how='left', on=['ID', 'UID', 'Vulnerable'])

    # Save only 
    correct = inspected.loc[inspected['Label'] == 1]
    correct.to_csv(filepath, mode='a', index=False, header=False)
    print(f'{len(correct)} vulnerable samples written to {filepath} from {dataset}')

if __name__ == '__main__':
    script = sys.argv[1]
    execute = {
        'save_validated_vulnerabilities': save_validated_vulnerabilities,
    }
    execute[script](sys.argv)