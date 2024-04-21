"""
Analyse duplicates in a dataset.
Consider near and exact matching.
"""
import sys
import csv
import ast
import gzip
import subprocess
import pandas as pd
from dq_analysis.datasets.data import Data
from ast import literal_eval

if len(sys.argv) < 2:
    print("Usage: python dq_analysis/attributes/uniqueness.py [prepare/measure] <dataset>")
    exit()


def identify_duplicates(dataset):
    """
    Identify near duplicates using the duplicate code detector tool
      from Allamanis (2018)

    IMPORTANT: Requires tokens to be generated first via currency preparation
    """

    csv.field_size_limit(100000000)
    headers = []
    data = []

    # Process tokenized files
    for row in csv.reader(open(f'dq_analysis/datasets/{dataset}/tokens.csv')):
        if not headers:
            headers = row
        else:
            entries = [a[0] for a in ast.literal_eval(row[3])]
            data.append( [row[1], entries])
    df = pd.DataFrame(data, columns=['filename','tokens'])

    # Output in JSONL format
    df.to_json(f'dq_analysis/datasets/{dataset}/tokens.jsonl', orient='records', lines=True)
    # Output in GZIP format
    with open(f'dq_analysis/datasets/{dataset}/tokens.jsonl', 'rb') as src, gzip.open(f'dq_analysis/datasets/{dataset}/tokens.jsonl.gz', 'wb') as dst:
        dst.writelines(src)

    # Run duplicate detector tool
    p = subprocess.Popen(f"dotnet run DuplicateCodeDetector.csproj --dir=../../dq_analysis/datasets/{dataset}/tokens.jsonl.gz",
                         cwd='near-duplicate-code-detector/DuplicateCodeDetector/', shell=True)
    p.wait()

    # Move the output
    p = subprocess.Popen(f"mv near-duplicate-code-detector/DuplicateCodeDetector/DuplicateCodeDetector.csproj.json dq_analysis/datasets/{dataset}/unique_clusters.csv", shell=True)
    p.wait()


def get_duplicate_clusters(dataset, type=3):
    """
    Return the within-class fuzzy duplicates of a dataset,
        using similarity matching.
    """

    # Load the data
    data = Data(dataset).get_dataset()
    vuln = data[data.Vulnerable == 1].UID.tolist()
    nonvuln = data[data.Vulnerable == 0].UID.tolist()

    # Read near duplicate matching output
    if type == 1:
        duplicates = open(f'dq_analysis/datasets/{dataset}/consistent_clusters.csv')
        clusters = literal_eval(duplicates.read())
    elif type == 3:
        duplicates = open(f'dq_analysis/datasets/{dataset}/unique_clusters.csv')
        clusters = literal_eval(duplicates.read())
    class_clusters = []
    # Split clusters by class
    for x in clusters:
        cluster0, cluster1 = [], []
        for id in x:
            if int(id) in nonvuln:
                cluster0.append(int(id))
            if int(id) in vuln:
                cluster1.append(int(id))
        if len(cluster0) > 1:
            class_clusters.append(cluster0)
        if len(cluster1) > 1:
            class_clusters.append(cluster1)
    return class_clusters


def count_near_unique(dataset, type):
    """
    Count number of unique files using near duplicate matching,
        performed using the Jacquard Index and implemented via Allamanis (2018)
    """
    print('-'*3 + dataset + ' Type ' + str(type) + '-'*3)
    df = Data(dataset).get_dataset()
    # Get duplicates
    class_clusters = get_duplicate_clusters(dataset, type)
    duplicates = [int(y) for x in class_clusters for y in x[1:]]
    # Get unique
    unique = df[~df.UID.isin(duplicates)]
    unique = unique.dropna()
    num_unique = len(unique)

    print(f"NEAR unique: {num_unique} / {len(df)}")
    print(f"{dataset} Uniqueness = {num_unique / len(df)}")


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        identify_duplicates(sys.argv[2])
    elif sys.argv[1] == 'measure':
        count_near_unique(sys.argv[2])
    else:
        print(f"ERROR: Unknown command line argument: \"{sys.argv[1]}\"")
