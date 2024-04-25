import sys
import os.path
import pandas as pd
import argparse
import json
import logging
import numpy as np
from dq_analysis.datasets.data import Data, KNOWN_DATASETS
from dq_analysis.attributes.consistency import get_consistent_dataset
from openai import OpenAI
from sklearn.metrics import f1_score

VALIDATED_SAMPLE_PATH = 'dq_analysis/datasets/all_validated.csv'
NEEDED_FIELDS = ['ID', 'Vulnerable', 'Function']

def save_validated_vulns(dataset, filepath='', overwrite=False, max=99999999, min=0, add_safe_samples=False):
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

    # Load manually inspected samples, save only correctly labeled vulnerable samples
    inspected = pd.read_csv(samplepath)
    correct = inspected.loc[inspected['Label'] == 1]
    correct.reindex(['UID', 'ID', 'Vulnerable'], axis=1)
    correct = pd.merge(correct,data, on=['UID', 'ID', 'Vulnerable'], how='left')
    correct = correct[NEEDED_FIELDS]

    # filter the test set to the same max and min code length as training data
    correct = correct.loc[correct['Function'].str.len() < max]
    correct = correct.loc[correct['Function'].str.len() > min]
    num_vulns = len(correct)
    num_safe = 0

    if add_safe_samples:
        # Add safe samples in equal number to vulnerable samples
        # shuffle rows
        # filter the test set to the same max and min code length as training data
        data = data.loc[data['Function'].str.len() < max]
        data = data.loc[data['Function'].str.len() > min]
        data.sample(frac=1)
        safe = data.loc[data['Vulnerable'] == 0]
        safe = safe.iloc[:num_vulns]
        num_safe = len(safe)
        print(f'Length safe: {num_safe}, length vulnerable: {len(correct)}')
        print(correct.columns.values)
        correct = pd.merge(correct,safe[NEEDED_FIELDS], how='outer', on=NEEDED_FIELDS)
        print(f'num total: {len(correct)}')

    # add header if file doesn't exist yet or is being overwritten, otherwise append
    if filepath:
        if overwrite or not os.path.isfile(filepath):
            correct.to_csv(filepath, mode='w', index=False, header=True)
        else:
            existing = pd.read_csv(filepath, usecols=NEEDED_FIELDS)
            correct = correct.merge(existing, how='outer', on=NEEDED_FIELDS)
            correct.to_csv(filepath, mode='w', index=False, header=True)
        print(f'Wrote samples to {filepath}')
    
    print(f'Added {num_vulns} vulnerable samples and {num_safe} safe samples from {dataset}')
    
    return correct

def collect_test_set(filepath="results/custom_datasets/test_set.csv", datasets=KNOWN_DATASETS, max=99999999, min=0):
    """
    Cycle through all known datasets collecting validated vulnerability samples,
    write to a csv file

    Example:
    python scripts.py --script collect_test_set --output_filepath "results/custom_datasets/test_set.csv"
    """
    overwrite=True
    add_safe_samples = True
    test_set = save_validated_vulns(datasets[0], "", overwrite, max, min, add_safe_samples)
    overwrite=False
    for dataset in datasets[1:]:
        new_set = save_validated_vulns(dataset, "", overwrite, max, min, add_safe_samples)
        test_set = test_set.merge(new_set, how='outer', on=NEEDED_FIELDS)

    # Add vulchecker samples
    vulchecker_samples_path = 'results/custom_datasets/Vulchecker/'
    vulchecker_vulns = pd.read_csv(vulchecker_samples_path + 'vulchecker_samples_vulnerable.csv')[NEEDED_FIELDS]
    vulchecker_safe = pd.read_csv(vulchecker_samples_path + 'vulchecker_samples_safe.csv')[NEEDED_FIELDS]
    vulchecker = vulchecker_vulns.merge(vulchecker_safe, how='outer', on=NEEDED_FIELDS)
    vulchecker = vulchecker.loc[vulchecker['Function'].str.len() < max]
    vulchecker = vulchecker.loc[vulchecker['Function'].str.len() > min]
    print(f'Adding {len(vulchecker)} Vulchecker samples')
    test_set = test_set.merge(vulchecker, how='outer', on=NEEDED_FIELDS)
    test_set.sample(frac=1) # shuffle

    if (filepath):
        test_set.to_csv(filepath, mode='w', index=False, header=True)
        print(f'Saved test samples to {filepath}')

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
      --jsonl_structure=code:Function,idx:ID,target:Vulnerable
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

def make_jsonl_dataset(data_filepath, output_filepath, exclude_from_tests='', max=20000, min=0):
    """
    Example:
    python scripts.py --script make_jsonl_dataset --max 1024 
        --dataset_filepath='results/consistent_unique_datasets/Devign.csv' 
        --output_filepath='devign_filtered' --exclude_from_tests=Devign
    """
    jsonl_structure = {
        'code':'Function',
        'idx':'ID',
        'target':'Vulnerable',
    }

    # create the validated test set
    datasets = KNOWN_DATASETS
    if exclude_from_tests in datasets:
        datasets.remove(exclude_from_tests)
    test = collect_test_set("", datasets, max, min)
    test = test.iloc[np.random.permutation(len(test))]  # shuffle rows

    # load the dataset
    data = _filter_by_size(data_filepath, max, min)

    # Important! Make the dataset balanced
    data.iloc[np.random.permutation(len(data))]  # shuffle rows
    vulnerable_set = data.loc[data['Vulnerable'] == 1]
    safe_set = data.loc[data['Vulnerable'] == 0]
    if len(safe_set) > len(vulnerable_set):
        safe_set = safe_set.iloc[:len(vulnerable_set)]
    else:
        vulnerable_set = vulnerable_set.iloc[:len(safe_set)]
    # Combine equal sized safe and vulnerable sets
    data = vulnerable_set.merge(safe_set, how='outer', on=NEEDED_FIELDS)
    
    #split into train and eval sets
    if len(data) > 2500:
        category_size = 135
    else:
        category_size = len(data)//20
    
    # Collect equal number of safe and vulnerable samples
    data.iloc[np.random.permutation(len(data))]  # shuffle rows
    eval_vulnerable = data.loc[data['Vulnerable'] == 1].iloc[:category_size]
    eval_safe = data.loc[data['Vulnerable'] == 0].iloc[:category_size]
    eval = eval_vulnerable.merge(eval_safe, how='outer', on=NEEDED_FIELDS)
    eval.iloc[np.random.permutation(len(eval))]  # shuffle rows
    # flag the rows that are *only* in data and not also in eval
    eval_with_data = data.merge(eval, on=NEEDED_FIELDS, how='left', indicator=True)
    # Collect the rows that were *only* in data
    train = eval_with_data[eval_with_data['_merge'] == 'left_only']

    eval.iloc[np.random.permutation(len(eval))]  # shuffle rows
    train.iloc[np.random.permutation(len(train))]  # shuffle rows

    print(f'eval length: {len(eval)}, train length: {len(train)}')
    # save datasets
    export_to_jsonl("", f'{output_filepath}/train.jsonl', jsonl_structure, train)
    export_to_jsonl("", f'{output_filepath}/valid.jsonl', jsonl_structure, eval)
    export_to_jsonl("", f'{output_filepath}/test.jsonl', jsonl_structure, test)

    print(f'Wrote {len(train)} balanced training samples, {len(eval)} balanced evaluation samples,'+
        f'and {len(test)} balanced test samples to {output_filepath}')

def openai_fix_vulns(dataset_filepath, output_filepath):
    client = OpenAI()
    data = pd.read_csv(dataset_filepath)
    model="gpt-4-turbo"
    prompt_tokens = 0
    completion_tokens = 0
    role = "You are an amazing cyber security expert and skilled coder. " \
        "You get a list of security vulnerabilities and C code that contains one or " \
        "more of each vulnerability, which you will fix. Do NOT change what the " \
        "code does, or variable or function names, or add new comments, " \
        "and keep code changes succinct. But you MUST find and fix the vulnerability or vulnerabilities. " \
        "Only output a properly formatted JSON object! " \
        "The first field is 'analysis', with a brief description of " \
        "which lines have the vulnerabilities and how they will be fixed. " \
        "The second field is 'code', containing the fixed code. Do not truncate any code, all code must be returned. Do not change whitespace or escaped characters, and match the existing indentation."
    
    for i in range(0,len(data)):
        vulnerability_type = data.iloc[i]["Diagnosis"]
        code = data.iloc[i]["Function"]
        message = f"Vulnerability types: {vulnerability_type}\n\n{code}"
        max_attempts = 3
        attempts = 0
        completion = {}
        while attempts < max_attempts:
            attempts += 1
            try:
                completion = client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": role},
                        {"role": "user", "content": message}
                    ],
                    temperature=0.3, # max is 2, don't get creative, be correct
                    frequency_penalty=-0.1 # -2 to 2. Use negative value to encourage reuse of terms as we want code duplicated
                )
                result = json.loads(completion.choices[0].message.content)
                completion_tokens += completion.usage.completion_tokens
                prompt_tokens += completion.usage.prompt_tokens
                print(f'***Item {i}, {data.iloc[i]["ID"]} ({completion.usage})***')
                print(message)
                print(result["analysis"])
                print(result["code"])
                data.at[i, "Function"] = result["code"]
                attempts = max_attempts
            except Exception as e:
                print(f'Failed Item {i}, {data.iloc[i]["ID"]} attempt #{attempts}')
                print(f'completion object: {completion}')
                print(repr(e))
    
    data.to_csv(output_filepath, mode='w', index=False, header=True)
    print(f'Task complete, {len(data)} functions written to {output_filepath}, {prompt_tokens} ' +
          f'prompt tokens used, {completion_tokens} completion tokens used with {model}')
    print(f'Example of response structure, should include exact model used: {completion}')

def openai_run_tests(test_filepath, output_filepath, model="gpt-3.5-turbo", role=""):
    """
    Example:
    python scripts.py --script openai_run_tests\
        --dataset_filepath='results/custom_datasets/test_set.csv'\
        --output_filepath='results/testing_runs/gpt-4-turbo_2024-04-25-03'\
        --model='gpt-3.5-turbo'\
        --model_role="You are an experienced cyber security expert and skilled coder. "\
"A function written in C is provided, it may contain one of the top "\
"Mitre CWE vulnerabilities, it may not. You will carefully inspect the code "\
"and determine if the code has a serious vulnerability. "\
"Only output a properly formatted JSON object! "\
"The first field is 'analysis', you will provide a careful analysis of whether or not this code "\
"has a serious vulnerability. If it has a vulnerability, has it "\
"already been mitigated in the code? "\
"The second field is 'vulnerable', it is a binary field. It must be "\
"either 1 if the code has a serious vulnerability, "\
"or 0 if it doesn't."
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(output_filepath)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    client = OpenAI()
    data = pd.read_csv(test_filepath)
    data = data.sample(frac=1) # shuffle
    prompt_tokens = 0
    completion_tokens = 0
    sample_vulnerability = []
    predicted_vulnerability = []
    results = {
        'tp': 0,
        'tn': 0,
        'fp': 0,
        'fn': 0,
    }
    logger.info(f'The role is {role}, the prompt is just the code')

    for i in range(0,len(data)):
        code = data.iloc[i]["Function"]
        vulnerable = int(data.iloc[i]["Vulnerable"])
        max_attempts = 3
        attempts = 0
        completion = {}
        while attempts < max_attempts:
            attempts += 1
            try:
                completion = client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": role},
                        {"role": "user", "content": code}
                    ],
                    temperature=0.3, # max is 2, don't get creative, be correct
                )
                result = json.loads(completion.choices[0].message.content)
                completion_tokens += completion.usage.completion_tokens
                prompt_tokens += completion.usage.prompt_tokens
                logger.info(f'***Item {i}, {data.iloc[i]["ID"]} ({completion.usage})***')
                logger.info(f'******* {vulnerable}, {code}')
                logger.info(result)
                predicted = int(result['vulnerable'])

                # Record the result
                if predicted != 1 and predicted != 0:
                    raise ValueError(f'Result must not have been properly formatted. ' \
                                     f'predicted is {predicted}, completion is {completion}')

                logger.info(f'For {data.iloc[i]["ID"]} the predicted is {predicted}, the value is supposed ' \
                      f'to be {vulnerable}')
                sample_vulnerability.append(vulnerable)
                predicted_vulnerability.append(predicted)
                logger.info(list(zip(sample_vulnerability, predicted_vulnerability)))
                if predicted == vulnerable and predicted == 1:
                    results['tp'] += 1
                elif predicted == vulnerable and predicted == 0:
                    results['tn'] += 1
                elif predicted != vulnerable and predicted == 1:
                    results['fp'] += 1
                elif predicted != vulnerable and predicted == 0:
                    results['fn'] +=1
                attempts = max_attempts
            except Exception as e:
                logger.error(f'Failed Item {i}, {data.iloc[i]["ID"]} attempt #{attempts}')
                logger.error(f'completion object: {completion}')
                logger.error(repr(e))
    
    logger.info(f'Example of response structure, should include exact model used: {completion}')
    logger.info(f'Task complete, {len(data)} functions written to {output_filepath}, {prompt_tokens} ' +
          f'prompt tokens used, {completion_tokens} completion tokens used with {model}')
    logger.info(f'Results: {list(zip(sample_vulnerability, predicted_vulnerability))}')
    logger.info(f'Detailed Results: {results}')
    logger.info(f'F1 score: {f1_score(sample_vulnerability, predicted_vulnerability)}')

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
    parser.add_argument("--exclude_from_tests", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--model_role", type=str, default="")

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    scripts= {
        'collect_test_set': collect_test_set,
        'save_validated_vulns': save_validated_vulns,
        'export_to_jsonl': export_to_jsonl,
        'intersection_consistent_unique': intersection_consistent_unique,
        'filter_by_size': filter_by_size,
        'openai_fix_vulns': openai_fix_vulns,
        'openai_run_tests': openai_run_tests,
    }


    if (args.script == 'collect_test_set'):
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
        make_jsonl_dataset(args.dataset_filepath, args.output_filepath, args.exclude_from_tests, args.max_length, args.min_length)
    elif (args.script == 'openai_fix_vulns'):
        if args.output_filepath == "":
            raise ValueError(f'--ouput_filepath must be included with {args.script}')
        if args.dataset_filepath == "":
            raise ValueError(f'--dataset_filepath must be included with {args.script}')
        openai_fix_vulns(args.dataset_filepath, args.output_filepath)
    elif (args.script == 'openai_run_tests'):
        openai_run_tests(args.dataset_filepath, args.output_filepath, args.model, args.model_role)
    else:
        raise ValueError(f'--script is {args.script}, but must be one of: {list(scripts.keys())}')

