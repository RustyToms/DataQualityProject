import sys
import os.path
import pandas as pd
import argparse
from dq_analysis.datasets.data import Data, KNOWN_DATASETS
from dq_analysis.attributes.consistency import get_consistent_dataset

VALIDATED_SAMPLE_PATH = 'dq_analysis/datasets/all_validated.csv'
NEEDED_FIELDS = ['ID', 'UID', 'Vulnerable', 'Function']

def save_validated_vulns(dataset, filepath='', overwrite=False, add_safe_samples=False):
    """
    Collect vulnerabilities that have been manually validated and referenced in sample.csv
    
    Example:
    python scripts.py --script save_validated_vulns --dataset Devign 
      --output_filepath=dq_analysis/datasets/devign_validated.csv --overwrite
    """
    samplepath = f'dq_analysis/datasets/{dataset}/sample.csv'

    if not os.path.isfile(samplepath):
        print(f'no file found at {samplepath} for {dataset} dataset')
        # Return empty dataframe with the proper columns
        return pd.DataFrame(columns=NEEDED_FIELDS)
    
    print(f'Collecting manually validated vulnerabilities from {dataset}')
    data = Data(dataset).get_dataset()

    # Load manually inspected samples
    inspected = pd.read_csv(samplepath)
    inspected = inspected.merge(data, how='left', on=['ID', 'UID', 'Vulnerable'])

    # Save only correctly labeled vulnerable samples
    correct = inspected.loc[inspected['Label'] == 1]
    correct = correct[NEEDED_FIELDS]
    num_vulns = len(correct)
    num_safe = 0

    if add_safe_samples:
        # Add safe samples in equal number to vulnerable samples
        # shuffle rows
        data.sample(frac=1)
        safe = data.loc[data['Vulnerable'] == 0]
        safe = safe.iloc[:len(correct)]
        num_safe = len(safe)
        correct = correct.merge(safe, how='outer', on=NEEDED_FIELDS)


    # add header if file doesn't exist yet or is being overwritten, otherwise append
    if filepath:
        if overwrite or not os.path.isfile(filepath):
            correct.to_csv(filepath, mode='w', index=False, header=True)
        else:
            existing = pd.read_csv(filepath)[NEEDED_FIELDS]
            correct = correct.merge(existing, how='outer', on=NEEDED_FIELDS)
            correct.to_csv(filepath, mode='w', index=False, header=True)
        print(f'Wrote samples to {filepath}')
    
    print(f'Added {num_vulns} vulnerable samples and {num_safe} safe samples from {dataset}')
    
    return correct

def collect_test_set(filepath=""):
    """
    Cycle through all known datasets collecting validated vulnerability samples,
    write to a csv file

    Example:
    python scripts.py --script collect_test_set
    """
    overwrite=True
    add_safe_samples = True
    test_set = save_validated_vulns(KNOWN_DATASETS[0], filepath, overwrite, add_safe_samples)
    overwrite=False
    for dataset in KNOWN_DATASETS[1:]:
        new_set = save_validated_vulns(dataset, filepath, overwrite, add_safe_samples)
        test_set = test_set.merge(new_set, how='outer', on=NEEDED_FIELDS)

    print(f'Created {len(test_set)} test samples')
    return test_set

def intersection_consistent_unique(dataset, filepath):
    """
    Combine the consistent and unique datasets found in results,
    for a certain dataset, and save it as a csv

    Example:
    python scripts.py --script intersection_consistent_unique --dataset 'Devign' 
      --output_filepath results/consistent_unique_datasets/Devign.csv
    """
    consistent = pd.read_csv(f'results/consistent_datasets/{dataset}.csv')[NEEDED_FIELDS]
    unique = pd.read_csv(f'results/unique_datasets/{dataset}.csv')[NEEDED_FIELDS]
    intersection = pd.merge(consistent, unique, how='inner', on=NEEDED_FIELDS)
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

    # create the validated test set            
    test = collect_test_set()

    # filter the test set to the same max and min code length as training data
    test_length = len(test)
    test = test.loc[test['Function'].str.len() < max]
    test = test.loc[test['Function'].str.len() > min]
    print(f'Test samples filtered to be under {max} and over {min} characters. '+
          f'{test_length} samples reduced to {len(test)} samples.')
    
    # load the dataset
    data = _filter_by_size(data_filepath, max, min)
    # VERY IMPORTANT!! Filter out the test samples that are in the training data
    # The below code, in combination with the max/min filtering, ensures that
    # no test samples are from the same project as the training samples.
    before = len(test)
    # flag the rows that are *only* in data and not also in test
    tests_with_data = test.merge(data, on=['UID', 'Vulnerable', 'Function'], how='left', indicator=True)
    # Collect the rows that were *only* in data
    test = tests_with_data[tests_with_data['_merge'] == 'left_only']
    after = len(test)
    print(f'{before-after} test samples removed that were in the data set, {after} remaining')

    # shuffle rows
    test.sample(frac=1)
    data.sample(frac=1)
    #split into train and eval sets
    if len(data) > 2500:
        eval_length = 270
    else:
        eval_length = len(data)//10

    eval = data.iloc[:eval_length]
    train = data.iloc[eval_length:]
    print(f'eval length: {len(eval)}, train length: {len(train)}')
    # save datasets
    export_to_jsonl("", f'{output_filepath}/train.jsonl', jsonl_structure, train)
    export_to_jsonl("", f'{output_filepath}/valid.jsonl', jsonl_structure, eval)
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
        'collect_test_set': collect_test_set,
        'save_validated_vulns': save_validated_vulns,
        'export_to_jsonl': export_to_jsonl,
        'intersection_consistent_unique': intersection_consistent_unique,
        'filter_by_size': filter_by_size,
    }


    if (args.script == 'collect_test_set'):
        if args.output_filepath == "":
            raise ValueError(f'--ouput_filepath must be included with {args.script}')
        collect_test_set(args.output_filepath)
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

