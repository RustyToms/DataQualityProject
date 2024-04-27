import sys
import os.path
import pandas as pd
import argparse
import json
import logging
import time
import csv
import re
import numpy as np
from dq_analysis.datasets.data import Data, KNOWN_DATASETS
from dq_analysis.attributes.consistency import get_consistent_dataset
from openai import OpenAI
from sklearn.metrics import f1_score

VALIDATED_SAMPLE_PATH = 'dq_analysis/datasets/all_validated.csv'
NEEDED_FIELDS = ['ID', 'Vulnerable', 'Function']
DANGEROUS_CWES = [
    {'id': 'CWE-787', 'description': 'Out-of-bounds write'},
    {'id': 'CWE-79', 'description': 'Cross-site scripting'},
    {'id': 'CWE-89', 'description': 'SQL injection'},
    {'id': 'CWE-416', 'description': 'Use after free'},
    {'id': 'CWE-78', 'description': 'OS command injection'},
    {'id': 'CWE-20', 'description': 'Improper input validation'},
    {'id': 'CWE-125', 'description': 'Out-of-bounds read'},
    {'id': 'CWE-22', 'description': 'Path traversal'},
    {'id': 'CWE-352', 'description': 'Cross-site request forgery (CSRF)'},
    {'id': 'CWE-434', 'description': 'Unrestricted upload of file with dangerous type'},
    {'id': 'CWE-862', 'description': 'Missing authorization'},
    {'id': 'CWE-476', 'description': 'NULL pointer dereference'},
    {'id': 'CWE-287', 'description': 'Improper authentication'},
    {'id': 'CWE-190', 'description': 'Integer Overflow or wraparound'},
    {'id': 'CWE-502', 'description': 'Deserialization of untrusted data'},
    {'id': 'CWE-77', 'description': 'Command injection'},
    {'id': 'CWE-119', 'description': 'Improper restriction of operations within the bounds of a memory buffer'},
    {'id': 'CWE-798', 'description': 'Useof hard-coded credentials'},
    {'id': 'CWE-918', 'description': 'Server-side request forgery (SSRF)'},
    {'id': 'CWE-306', 'description': 'Missing authentication for critical function'},
    {'id': 'CWE-362', 'description': 'Race condition'},
    {'id': 'CWE-269', 'description': 'Improper privilege management'},
    {'id': 'CWE-94', 'description': 'Code injection'},
    {'id': 'CWE-863', 'description': 'Incorrect authorization'},
    {'id': 'CWE-276', 'description': 'Incorrect default permissions'},
    {'id': 'CWE-122', 'description': 'Buffer overflow'},
    {'id': 'CWE-590', 'description': 'Free of memory not on the heap'},
    {'id': 'CWE-242', 'description': 'Use of inherently dangerous function'},
    {'id': 'CWE-789', 'description': 'Memory allocation with excessive size value (stack exhaustion)'},
    {'id': 'CWE-1341', 'description': 'Multiple release of same resource'},
    {'id': 'CWE-672', 'description': 'Operation on a resource after expiration or release'},
    {'id': 'CWE-189', 'description': 'Numeric errors'},
    {'id': 'CWE-200', 'description': 'Exposure of sensitive information to an unauthorized actor'},
    {'id': 'CWE-254', 'description': '7PK security features'},
    {'id': 'CWE-264', 'description': 'Permission, privileges, and access controls'},
    {'id': 'CWE-284', 'description': 'Improper access control'},
    {'id': 'CWE-399', 'description': 'Resource management errors'},
    {'id': 'CWE-834', 'description': 'Excessive iteration'},
    {'id': 'CWE-843', 'description': 'Type confusion'},
]

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
    python scripts.py --script export_to_jsonl --output_filepath results/custom_datasets/test_set_clean9.jsonl --dataset_filepath results/custom_datasets/test_set_clean9.csv --jsonl_structure=code:Function,idx:ID,target:Vulnerable
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

def _normalize_c_code(code):
    # Remove trailing whitespaces
    code = re.sub(r'[ \t]+$', '', code, flags=re.MULTILINE)
    # Reduce spaces around operators and punctuation, except within string literals
    code = re.sub(r'(?<!["\'])\s*([{}();,])\s*', r'\1', code)
    code = re.sub(r'\s*([+\-*/=<>&|!%])\s*', r'\1', code)
    # Normalize newlines: remove extra newlines, keep only one
    code = re.sub(r'\n\s*\n', '\n', code)
    # Minimize spaces around assignment and comparison operators
    code = re.sub(r'\s*([=<>!]=|&&|\|\|)\s*', r' \1 ', code)
    # Ensure one space after commas in function calls/declarations, no space before
    code = re.sub(r',\s*', ', ', code)
    return code

def _openai_modify_code(role, code, model, logger):
    client = OpenAI()
    attempts = 0
    error_count = 0
    max_errors = 1
    max_attempts = 3
    changed_code = ''
    prompt_tokens = 0
    response_tokens = 0
    normalized_code = _normalize_c_code(code)

    while attempts < max_attempts and error_count < max_errors:
        attempts += 1
        changed_code = ''
        try:
            completion = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": role},
                    {"role": "user", "content": normalized_code}
                ],
                temperature=0.3, # max is 2, don't get creative, be correct
                frequency_penalty=-0.1, # -2 to 2. Use negative value to encourage reuse of terms as we want code duplicated
                logit_bias={
                    "1734": -50, # try to stop the infinite \n glitch, different than actual new line
                    "5061": -50, # try to stop the infinite \t glitch, different than actual tab
                },
            )
            result = json.loads(completion.choices[0].message.content)
            prompt_tokens = completion.usage.prompt_tokens
            response_tokens = completion.usage.completion_tokens

            logger.info(f'***({completion.usage} {completion.model})***')
            logger.info(code)
            logger.info(result["analysis"])
            changed_code = result["code"]
            # check if the code is of similar length
            normalized_changed_code = _normalize_c_code(changed_code)
            delta = abs(len(normalized_changed_code) - len(normalized_code))
            percent = delta / len(normalized_code)
            logger.info(f'Modified code:\n{changed_code}')
            if percent > 0.1 and delta > 110:                
                logger.error(f'Modified code length is too different, {len(normalized_changed_code)} chars after whitespace removal vs original {len(normalized_code)} chars after whitespace removal')
                changed_code = ''
                continue
            else:
                logger.info(f'Code modified, {len(normalized_changed_code)} chars after whitespace removal vs original {len(normalized_code)} chars after whitespace removal')
            break
        except Exception as e:
            error_count += 1
            changed_code = ''
            logger.error(f'Failed to change code, attempt #{attempts}')
            logger.error(f'completion object: {completion}')
            logger.error(repr(e))
    
    if not changed_code:
        logger.info('---------Unable to change code, returning original code----------------')
        changed_code = code

    return {
        'code': changed_code,
        'prompt_tokens': prompt_tokens,
        'response_tokens': response_tokens,
    }

def openai_fix_vulns(dataset_filepath, output_filepath, model, max=9999999, min=0):
    """
    Example:
    python scripts.py --script openai_fix_vulns\
        --dataset_filepath='results/custom_datasets/test_set.csv'\
        --output_filepath='results/custom_datasets/test_set_clean.csv'\
        --model='gpt-4-turbo' --max 2048 --min 0
    """
    role = "You are an elite cyber security expert and coder. A C function is provided. You will ensure it has no security vulnerabilities, and fix any you find. Do NOT change what the code does, or variable or function names. Don't add new comments. Keep code changes succinct. But find and fix any vulnerabilities you find. Only output a properly formatted JSON object! The first field is 'analysis', with a very brief description of any vulnerabilities and how they will be fixed. The second field is 'code', containing the fixed code. Do not truncate any code, all code must be returned. Do not change whitespace or escaped characters, do not replace spaces with tabs or tabs with spaces, match the existing indentation.  Except do not return more than 4 '\\n' or '\\t' characters, or any other non-whitespace token in a row!"
    data = _filter_by_size(dataset_filepath, 2048, 0)
    # turn into a list of dicts
    data_l = data.to_dict(orient='records')
    logger = _make_logger(logging.INFO, output_filepath+'-logs.txt')
    prompt_tokens = 0
    response_tokens = 0    

    for i in range(0,len(data_l)):
        if data_l[i]['Vulnerable'] == 0:
            code = data_l[i]["Function"]
            clean_code_dict = _openai_modify_code(role, code, model, logger)
            data_l[i]["Function"] = clean_code_dict['code']
            prompt_tokens += clean_code_dict['prompt_tokens']
            response_tokens += clean_code_dict['response_tokens']

    data = pd.DataFrame(data_l)
    data.to_csv(output_filepath, mode='w', index=False, header=True)
    logger.info(f'Task complete, {len(data)} functions written to {output_filepath}, cost {prompt_tokens} prompt tokens and {response_tokens} response tokens {model}')

def _make_logger(level, output_filepath=''):
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    if output_filepath:
        file_handler = logging.FileHandler(output_filepath)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

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
    logger = _make_logger(logging.INFO, output_filepath)
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

def openai_make_vulns(source, target, model, key, max=99999999, min=0):
    """
    Load dataset of non-vulnerable code
    submit to chatgpt with list of desired vulnerabilities
    Select appropriate vulnerability for this code
    Have chatgpt clean the code
    Save clean code, have chatgpt inject the vulnerability and save vulnerable code
    Keep track of how many of each vulnerability is used, try to balance
    No need to use pandas for this

    Example:
    python scripts.py --script openai_make_vulns --dataset_filepath='dq_analysis/datasets/ReVeal/non-vulnerables.json' --output_filepath='results/custom_datasets/synthetic/gpt4_reveal.jsonl' --model='gpt-4-turbo' --max_length 1600 --min_length 60  --key code
    """
    logger = _make_logger(logging.INFO, target + '-logs.txt')
    # create a dict with CWE ids as keys matched with their description
    # create a dict with CWE ids as keys to keep track of samples
    vuln_types = {}
    vuln_counts = {}
    for vuln in DANGEROUS_CWES:
        vuln_counts[vuln['id']] = 0
        vuln_types[vuln['id']] = vuln['id'] + ' ' + vuln['description']
    filetype = re.search('\w+$', source)[0]
    client = OpenAI()
    prompt_tokens = 0
    response_tokens = 0
    num_success = 0
    max_of_type = 20
    maxed_vulns = []

    if filetype == 'json':
        with open(source, "r") as f:
            source_list = json.load(f)
    elif filetype == 'jsonl':
        code_list = []
        with open(source, "r") as f:
            _source_list = list(f)
        for json_dict in _source_list:
            source_list.append(json.loads(json_dict))
    elif filetype == 'csv':
        with open(source, "r") as f:
            source_list = list(csv.DictReader(f))
    else:
        raise ValueError('filetype must be ".csv", ".json", or ".jsonl"')

    # normalize and filter by code length
    code_list = []
    for item in source_list:
        code = _normalize_c_code(item[key]) # remove whitespaces
        if len(code) > min and len(code) < max:
            code_list.append(code)

    sample_index = -1
    logger.info(f'Beginning calls to OpenAI with {len(code_list)} samples')
    for sample in code_list:
        # the list of vulnerabilities will change as we get enough examples of specific vulnerabilties
        role = f"You are an elite cyber security expert and coder. You are creating vulnerabilities in C functions to use in a dataset to train a cybersecurity model. A function written in C is provided. Analyze the code and determine which, if any of a list of vulnerabilies could be introduced into the code with minimal code changes. Only return a properly formatted JSON object! There will be two fields. The first will be 'analysis' with a 1-2 sentence explanation of your choice. The second will be 'vulnerability' and it will include only a CWE identifier, like 'CWE-000'. But if no vulnerability will work for this code, return 'None' instead of a CWE. \nPotential vulnerabilities: {list(vuln_types.keys())}"
        attempts = 0
        time_in_ms = int(time.time() * 1000)
        max_attempts = 1
        vuln_code = ''
        clean_code = ''
        sample_index += 1
        
        while attempts < max_attempts:
            attempts += 1
            vuln_code = ''
            clean_code = ''

            # Step 1: determine which vulnerability can be used
            try:
                completion = client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": role},
                        {"role": "user", "content": sample}
                    ],
                    temperature=0.4, # max is 2, don't get creative, be correct
                )
                result = json.loads(completion.choices[0].message.content)
                prompt_tokens += completion.usage.prompt_tokens
                response_tokens += completion.usage.completion_tokens
                vuln_type = result["vulnerability"]
                
                # move to the next code sample if no vulnerability selected
                if vuln_type.lower() == 'none':
                    break

                # Throw error if CWE is not formatted correctly or in the list
                if not vuln_type in vuln_types:
                    raise ValueError(f'"{vuln_type}" is not in the vuln_types list')

                logger.info(f'Index: {sample_index}, vulnerability: {vuln_type}, analysis: {result["analysis"]}\n({completion.usage})')
            except Exception as e:
                logger.error(f'Failure on attempt #attempt #{attempts} to analyze code sample for appropriate vulnerability at sample index {sample_index}')
                logger.error(f'Code sample:\n{sample}')
                logger.error(f'completion object: {completion}')
                logger.error(repr(e))
                # go to next iteration of the while loop, do not go to Step 2
                continue

            # Step 2: Save a clean version of the code
            try:
                clean_role = "You are an elite cyber security expert and coder. A C function is provided. You will ensure it has no security vulnerabilities, and fix any you find. Do NOT change what the code does, or variable or function names. Don't add new comments. Keep code changes succinct. But fix any vulnerabilities you find. Only output a properly formatted JSON object! The first field is 'analysis', with a very brief description of any vulnerabilities and how they will be fixed. The second field is 'code', containing the fixed code. Do not truncate any code, all code must be returned. Do not change whitespace or escaped characters, do not replace spaces with tabs or tabs with spaces, match the existing indentation.  Except do not return more than 4 consecutive '\\n' or '\\t' characters, or any other non-whitespace token! Do not remove comments. Example response: {{'analysis': 'Analysis goes here', 'code': 'code goes here'}}"
                # _openai_modify_code has code to check obvious mistakes and retry
                # Returns the original code if it needs to retry to many times
                clean_code_dict = _openai_modify_code(clean_role, sample, model, logger)
                clean_code = clean_code_dict['code']
                prompt_tokens += clean_code_dict['prompt_tokens']
                response_tokens += clean_code_dict['response_tokens']
                safe_sample = {
                    'idx': f'{vuln_type}-safe-{time_in_ms}',
                    'code': clean_code,
                    'target': 0,
                }

            except Exception as e:
                logger.error(f'Failure on attempt #{attempts} to clean code sample at sample index {sample_index}')
                logger.error(f'Code sample:\n{sample}')
                logger.error(repr(e))
                # go to next iteration of the while loop, do not go to Step 3
                continue

            # Step 3: Create a vulnerable version of the code
            try:
                vuln_role = f"You are an elite cyber security expert and coder. You are creating vulnerabilities in C functions to use in a dataset to train a cybersecurity model. A function written in C is provided. Modify this function to include: {vuln_types[vuln_type]}. Do NOT change what the code does, or variable or function names, or add new comments, and keep code changes succinct. Only return a properly formatted JSON object. There will be two fields. The first will be 'analysis' with a 1-2 sentence explanation of how you will insert the vulnerability. The second will be 'code' and it will include the changed code. Do not truncate any code, all code must be returned. Do not change whitespace or escaped characters, and match the existing indentation. Except do not return more than three '\\n' or '\\t' characters, or any other non-whitespace token in a row! Example response: {{'analysis': 'Analysis goes here', 'code': 'code goes here'}}"
                vuln_code_dict = _openai_modify_code(vuln_role, clean_code, model, logger)
                vuln_code = vuln_code_dict['code']
                prompt_tokens += vuln_code_dict['prompt_tokens']
                response_tokens += vuln_code_dict['response_tokens']

                vuln_sample = {
                    'idx': f'{vuln_type}-vuln-{time_in_ms}',
                    'code': vuln_code,
                    'target': 1,
                }
                logger.info(f'***Vulnerability generation complete ({completion.usage} {completion.model})***')
                logger.info(result["analysis"])
                logger.info(f'Vulnerable code:\n{vuln_code}')

            except Exception as e:
                logger.error(f'Failed to inject vulnerability, attempt #{attempts}')
                logger.error(f'completion object: {completion}')
                logger.error(repr(e))
                # go to next iteration of the while loop, do not save results
                continue

        if clean_code and vuln_code:
            vuln_counts[vuln_type]+=1
            if vuln_counts[vuln_type] >= max_of_type:
                maxed_vulns.append(vuln_type)
                vuln_counts.pop(vuln_type)
                vuln_types.pop(vuln_type)

        # append to jsonl file
        with open(target, 'a') as f:
            f.write(json.dumps(safe_sample) + "\n")
            f.write(json.dumps(vuln_sample) + "\n")   

        num_success += 1
        logger.info(f'Added another code pair at sample index {sample_index}. Created {num_success} safe/vulnerable code pairs and used {prompt_tokens} prompt tokens and {response_tokens} response tokens so far.')

        if len(vuln_types.keys()) < 6:
            break;
    
    # All done, report
    logger.info(f'Completed generating the dataset at sample index {sample_index} out of {len(code_list)} samples. Created {num_success} safe/vulnerable code pairs, saved at {target}. Used {prompt_tokens} prompt tokens and {response_tokens} response tokens')
    logger.info(f'{vuln_counts}\nGenerated {max_of_type} versions of the following: {maxed_vulns}')


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
    parser.add_argument("--key", type=str, default="")

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
        'openai_make_vulns': openai_make_vulns,
    }


    if (args.script == 'collect_test_set'):
        collect_test_set(args.output_filepath)
    elif (args.script == 'save_validated_vulns'):
        if args.output_filepath == "":
            raise ValueError(f'--output_filepath must be included with {args.script}')
        if args.dataset == "":
            raise ValueError(f'--dataset must be included with {args.script}')
        save_validated_vulns(args.dataset, args.output_filepath, args.overwrite)
    elif (args.script == 'export_to_jsonl'):
        if args.output_filepath == "":
            raise ValueError(f'--output_filepath must be included with {args.script}')
        if args.dataset_filepath == "":
            raise ValueError(f'--dataset_filepath must be included with {args.script}')
        if not bool(args.jsonl_structure):
            raise ValueError(f'--jsonl_structure must be included with {args.script}, '+
                             'as comma-separated json_key:pandas.DataFrame columns pairs, '+
                              'e.g. code:Function,idx:ID,status:Vulnerable')
        export_to_jsonl(args.dataset_filepath, args.output_filepath, args.jsonl_structure)
    elif (args.script == 'intersection_consistent_unique'):
        if args.output_filepath == "":
            raise ValueError(f'--output_filepath must be included with {args.script}')
        if args.dataset == "":
            raise ValueError(f'--dataset must be included with {args.script}')
        intersection_consistent_unique(args.dataset, args.output_filepath)
    elif (args.script == 'filter_by_size'):
        if args.output_filepath == "":
            raise ValueError(f'--output_filepath must be included with {args.script}')
        if args.dataset_filepath == "":
            raise ValueError(f'--dataset_filepath must be included with {args.script}')
        if args.max_length == 0:
            raise ValueError(f'--max_length must be included with {args.script}')
        filter_by_size(args.dataset_filepath, args.output_filepath, args.max_length, args.min_length)
    elif (args.script == 'make_jsonl_dataset'):
        if args.output_filepath == "":
            raise ValueError(f'--output_filepath must be included with {args.script}')
        if args.dataset_filepath == "":
            raise ValueError(f'--dataset_filepath must be included with {args.script}')
        make_jsonl_dataset(args.dataset_filepath, args.output_filepath, args.exclude_from_tests, args.max_length, args.min_length)
    elif (args.script == 'openai_fix_vulns'):
        if args.output_filepath == "":
            raise ValueError(f'--output_filepath must be included with {args.script}')
        if args.dataset_filepath == "":
            raise ValueError(f'--dataset_filepath must be included with {args.script}')
        if args.model == "":
            raise ValueError(f'--model must be included with {args.script}')
        openai_fix_vulns(args.dataset_filepath, args.output_filepath, args.model, args.max_length, args.min_length)
    elif (args.script == 'openai_run_tests'):
        openai_run_tests(args.dataset_filepath, args.output_filepath, args.model, args.model_role)
    elif(args.script == 'openai_make_vulns'):
        if args.output_filepath == "":
            raise ValueError(f'--output_filepath must be included with {args.script}')
        if args.dataset_filepath == "":
            raise ValueError(f'--dataset_filepath must be included with {args.script}')
        if args.model == "":
            raise ValueError(f'--model must be included with {args.script}')
        if args.key == "":
            raise ValueError(f'--key must be included with {args.script}, to find the code in the file')
        openai_make_vulns(args.dataset_filepath, args.output_filepath, args.model, args.key, args.max_length, args.min_length)
    else:
        raise ValueError(f'--script is {args.script}, but must be one of: {list(scripts.keys())}')

