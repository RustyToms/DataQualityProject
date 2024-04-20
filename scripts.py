import sys
import os.path
import pandas as pd
import argparse
from dq_analysis.datasets.data import Data, KNOWN_DATASETS
from dq_analysis.attributes.consistency import get_consistent_dataset

VALIDATED_SAMPLE_PATH = 'dq_analysis/datasets/all_validated.csv'

def save_validated_vulns(dataset, filepath, overwrite=False):
    """
    Collect vulnerabilities that have been manually validated and referenced in sample.csv
    
    Example:
    python scripts.py --script save_validated_vulns --dataset Devign 
      --output_filepath=dq_analysis/datasets/devign_validated.csv --overwrite
    """
    samplepath = f'dq_analysis/datasets/{dataset}/sample.csv'
    desired_fields = ['ID', 'UID', 'Vulnerable', 'Function']

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
    correct = correct[desired_fields]

    # add header if file doesn't exist yet or is being overwritten, otherwise append
    if overwrite or not os.path.isfile(samplepath):
        correct.to_csv(filepath, mode='w', index=False, header=True)
    else:
        existing = pd.read_csv(filepath)[desired_fields]
        correct = correct.merge(existing, how='outer', on=desired_fields)
        correct.to_csv(filepath, mode='w', index=False, header=True)
    
    print(f'{len(correct)} vulnerable samples written to {filepath} from {dataset}')

def collect_all_validated_vulnerabilities(filepath=VALIDATED_SAMPLE_PATH):
    """
    Cycle through all known datasets collecting validated vulnerability samples,
    write to a csv file

    Example:

    """
    is_first = True
    for dataset in KNOWN_DATASETS:
        save_validated_vulns(dataset, filepath, is_first)
        is_first=False
    file = pd.read_csv(filepath)
    print(f'Saved {len(file)} vulnerable samples to {filepath}')

def intersection_consistent_unique(dataset, filepath):
    """
    Combine the consistent and unique datasets found in results,
    for a certain dataset, and save it as a csv

    Example:
    python scripts.py --script intersection_consistent_unique --dataset 'Devign' 
      --output_filepath results/consistent_unique_datasets/Devign.csv
    """
    desired_fields = ['UID', 'Vulnerable', 'Function']
    consistent = pd.read_csv(f'results/consistent_datasets/{dataset}.csv')[desired_fields]
    unique = pd.read_csv(f'results/unique_datasets/{dataset}.csv')[desired_fields]
    intersection = pd.merge(consistent, unique, how='inner', on=desired_fields)
    intersection.dropna(inplace=True)
    intersection.to_csv(filepath, mode='w', index=False, header=True)

    print(f'{len(consistent)} consistent samples found, {len(unique)} unique samples found, '+
          f'{len(intersection)} consistent and unique samples found, saved to {filepath}')

def export_to_jsonl(dataset_filepath, jsonl_filepath, jsonl_structure, dataset=None):
    """
    Convert a csv dataset to a jsonl dataset, converting columns to keys 
    as described in jsonl_structure
    
    Example:
    python scripts.py --script export_to_jsonl 
      --output_filepath results/consistent_unique_datasets/Devign.jsonl 
      --dataset_filepath results/consistent_unique_datasets/Devign.csv 
      --jsonl_structure=code:Function,idx:UID,target:Vulnerable
    """
    if isinstance(dataset, pd.DataFrame):
        data = dataset
    else:
        data = pd.read_csv(dataset_filepath)
    
    data2 = data[jsonl_structure.values()]
    # invert keys and values
    inv_structure = {v: k for k,v in jsonl_structure.items()}
    data2.rename(columns=inv_structure, inplace=True)
    data2.to_json(path_or_buf=jsonl_filepath, orient='records', lines=True)
    print(f'Exported {dataset_filepath} to {jsonl_filepath} in jsonl format,'+
          f' matching these keys to columns: {jsonl_structure}')

def _filter_by_size(data_filepath, max, min):
    data = pd.read_csv(data_filepath)
    start_length = len(data)
    filtered_data = data.loc[data['Function'].str.len() < max]
    length = len(filtered_data)

    filtered_data = filtered_data.loc[data['Function'].str.len() > min]
    min_length = len(filtered_data)
    print(f'{length} samples are below {max} characters, {start_length - length} '+
          f'are {max} characters or more. {length-min_length} samples are at or below '+
          f'{min} characters')
    return filtered_data

def filter_by_size(data_filepath, output_filepath, max, min):
    """
    Make a dataset that is filtered by function size

    Example:
    python scripts.py --script filter_by_size 
      --dataset_filepath=results/consistent_unique_datasets/Devign.csv 
      --output_filepath=filtered_devign.csv --min_length=20 --max_length=1024
    """
    data = _filter_by_size(data_filepath, max, min)
    data.to_csv(output_filepath, mode='w', index=False, header=True)
    print(f'Saved {len(data)} samples to {output_filepath}')

def make_jsonl_dataset(data_filepath, output_filepath, max=20000, min=0):
    """
    Example:
    python scripts.py --script make_jsonl_dataset --max 1024 
        --dataset_filepath='results/consistent_unique_datasets/Devign.csv' 
        --output_filepath='devign_filtered'
    """
    jsonl_structure = {
        'code':'Function',
        'idx':'UID',
        'target':'Vulnerable',
    }
    data = _filter_by_size(data_filepath, max, min)
    # shuffle rows
    data.sample(frac=1)
    #split into train and eval sets
    if len(data) > 2500:
        eval_length = 270
    else:
        eval_length = len(data)//10
    eval = data.iloc[:eval_length]
    train = data.iloc[eval_length:]
    print(f'eval length: {len(eval)}, train length: {len(train)}')
    # save train and evaluation datasets
    export_to_jsonl("", f'{output_filepath}/train.jsonl', jsonl_structure, train)
    export_to_jsonl("", f'{output_filepath}/valid.jsonl', jsonl_structure, eval)

    # get validated test set
    # create the validated test set if it doesn't exist yet
    if not os.path.isfile(VALIDATED_SAMPLE_PATH):
        collect_all_validated_vulnerabilities(VALIDATED_SAMPLE_PATH)
            
    test = pd.read_csv(VALIDATED_SAMPLE_PATH)
    export_to_jsonl("", f'{output_filepath}/test.jsonl', jsonl_structure, test)

    print(f'Wrote {len(train)} training samples, {len(eval)} evaluation samples,'+
        f'and {len(test)} test samples to {output_filepath}')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, default="")
    parser.add_argument("--output_filepath", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--dataset_filepath", type=str, default="")
    parser.add_argument("--jsonl_structure", type=lambda x: {k:v for k,v in (i.split(':') for i in x.split(','))},
    help='comma-separated json_key:csv_column pairs, e.g. code:Function,idx:ID,status:Vulnerable', default={})
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--max_length", type=int, default=0)
    parser.add_argument("--min_length", type=int, default=0)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    scripts= {
        'collect_all_validated_vulnerabilities': collect_all_validated_vulnerabilities,
        'save_validated_vulns': save_validated_vulns,
        'export_to_jsonl': export_to_jsonl,
        'intersection_consistent_unique': intersection_consistent_unique,
        'filter_by_size': filter_by_size,
    }


    if (args.script == 'collect_all_validated_vulnerabilities'):
        if len(args.output_filepath) > 0:
            collect_all_validated_vulnerabilities(args.output_filepath)
        else:
            collect_all_validated_vulnerabilities()
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
            raise ValueError(f'--jsonl_structure must be included with {args.script}, '+
                             'as comma-separated json_key:pandas.DataFrame columns pairs, '+
                              'e.g. code:Function,idx:ID,status:Vulnerable')
        export_to_jsonl(args.dataset_filepath, args.output_filepath, args.jsonl_structure)
    elif (args.script == 'intersection_consistent_unique'):
        if args.output_filepath == "":
            raise ValueError(f'--ouput_filepath must be included with {args.script}')
        if args.dataset == "":
            raise ValueError(f'--dataset must be included with {args.script}')
        intersection_consistent_unique(args.dataset, args.output_filepath)
    elif (args.script == 'filter_by_size'):
        if args.output_filepath == "":
            raise ValueError(f'--ouput_filepath must be included with {args.script}')
        if args.dataset_filepath == "":
            raise ValueError(f'--dataset_filepath must be included with {args.script}')
        if args.max_length == 0:
            raise ValueError(f'--max_length must be included with {args.script}')
        filter_by_size(args.dataset_filepath, args.output_filepath, args.max_length, args.min_length)
    elif (args.script == 'make_jsonl_dataset'):
        if args.output_filepath == "":
            raise ValueError(f'--ouput_filepath must be included with {args.script}')
        if args.dataset_filepath == "":
            raise ValueError(f'--dataset_filepath must be included with {args.script}')
        make_jsonl_dataset(args.dataset_filepath, args.output_filepath, args.max_length, args.min_length)
    else:
        raise ValueError(f'--script is {args.script}, but must be one of: {list(scripts.keys())}')

