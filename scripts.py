import sys
import os.path
import pandas as pd
import argparse
from dq_analysis.datasets.data import Data, KNOWN_DATASETS
from dq_analysis.attributes.consistency import get_consistent_dataset

def save_validated_vulns(dataset, filepath, overwrite=False):
    """
    Collect vulnerabilities that have been manually validated and referenced in sample.csv
    """
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

    # add header if file doesn't exist yet or is being overwritten, otherwise append
    if overwrite or not os.path.isfile(samplepath):
        correct.to_csv(filepath, mode='w', index=False, header=True)
    else:
        correct.to_csv(filepath, mode='a', index=False, header=False)
    
    print(f'{len(correct)} vulnerable samples written to {filepath} from {dataset}')

def collect_all_validated_vulnerabilties(filepath):
    """
    Cycle through all known datasets collecting validated vulnerability samples, write to a csv file
    """
    is_first = True
    for dataset in KNOWN_DATASETS:
        save_validated_vulns(dataset, filepath, is_first)
        is_first=False
    file = pd.read_csv(filepath)
    print(f'Saved {len(file)} vulnerable samples to {filepath}')

def intersection_consistent_unique(dataset, filepath):
    consistent = pd.read_csv(f'results/consistent_datasets/{dataset}.csv')
    unique = pd.read_csv(f'results/unique_datasets/{dataset}.csv')
    intersection = pd.merge(consistent, unique, how='inner', on=['UID'])
    intersection.dropna(inplace=True)
    intersection.to_csv(filepath, mode='w', index=False, header=True)
    
    print(f'{len(consistent)} consistent samples found, {len(unique)} unique samples found, {len(intersection)} consistent and unique samples found, saved to {filepath}')

def export_to_jsonl(dataset_filepath, jsonl_filepath, jsonl_structure):
    """
    Convert a csv dataset to a jsonl dataset, converting columns to keys as described in jsonl_structure
    """
    data =  pd.read_csv(dataset_filepath)
    data2 = data[jsonl_structure.values()]
    # invert keys and values
    inv_structure = {v: k for k,v in jsonl_structure.items()}
    data2.rename(columns=inv_structure, inplace=True)
    print(data2)
    data2.to_json(path_or_buf=jsonl_filepath, orient='records', lines=True)
    print(f'Exported {dataset_filepath} to {jsonl_filepath} in jsonl format, matching these keys to columns: {jsonl_structure}')



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, default="")
    parser.add_argument("--output_filepath", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--dataset_filepath", type=str, default="")
    parser.add_argument("--jsonl_structure", type=lambda x: {k:v for k,v in (i.split(':') for i in x.split(','))},
    help='comma-separated json_key:csv_column pairs, e.g. code:Function,idx:ID,status:Vulnerable', default={})
    parser.add_argument("--overwrite", action="store_true", default=False)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    scripts= {
        'collect_all_validated_vulnerabilties': collect_all_validated_vulnerabilties,
        'save_validated_vulns': save_validated_vulns,
        'export_to_jsonl': export_to_jsonl,
        'intersection_consistent_unique': intersection_consistent_unique,
    }


    if (args.script == 'collect_all_validated_vulnerabilties'):
        if args.output_filepath == "":
            raise ValueError(f'--ouput_filepath must be included with {args.script}')
        collect_all_validated_vulnerabilties(args.output_filepath)
    elif (args.script == 'save_validated_vulns'):
        if args.output_filepath == "":
            raise ValueError(f'--ouput_filepath must be included with {args.script}')
        if args.dataset == "":
            raise ValueError(f'--dataset must be included with {args.script}')
        save_validated_vulns(args.dataset, args.output_filepath, args.overwrite)
    elif (args.script == 'export_to_jsonl'):
        if args.output_filepath == "":
            raise ValueError(f'--ouput_filepath must be included with {args.script}')
        if args.dataset_filepath == "":
            raise ValueError(f'--dataset_filepath must be included with {args.script}')
        if not bool(args.jsonl_structure):
            raise ValueError(f'--jsonl_structure must be included with {args.script}, as comma-separated json_key:pandas.DataFrame columns pairs, e.g. code:Function,idx:ID,status:Vulnerable')
        export_to_jsonl(args.dataset_filepath, args.output_filepath, args.jsonl_structure)
    elif (args.script == 'intersection_consistent_unique'):
        if args.output_filepath == "":
            raise ValueError(f'--ouput_filepath must be included with {args.script}')
        if args.dataset == "":
            raise ValueError(f'--dataset must be included with {args.script}')
        intersection_consistent_unique(args.dataset, args.output_filepath)
    else:
        raise ValueError(f'--script is {args.script}, but must be one of: {list(scripts.keys())}')

