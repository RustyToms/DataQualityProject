"""
Analyse the currency of datasets.
"""
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from sctokenizer import CTokenizer
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import jensenshannon, cosine
from dq_analysis.datasets.data import Data
# from dq_analysis.svp.tsne import plot_embeddings

if len(sys.argv) < 2:
    print("Usage: python dq_analysis/attributes/currency.py [prepare/measure] <dataset>")
    exit()


def tokenize_dataset(dataset):
    """Tokenize the functions of a dataset."""

    # Initialize codebert tokenizer
    tokenizer = CTokenizer()
    # Read the dataset
    df = Data(dataset).get_dataset()

    # Extract token information
    def tokenize(text):
        # Note: Escape characters break tokenization
        tokens = tokenizer.tokenize(text.replace('\\\\', ''))
        tokens = [(i.token_value, i.token_type.name) for i in tokens]
        return tokens

    start = time.time()
    if 'Wild-C' not in dataset:
        df['Token'] = df['Function'].astype(str).apply(lambda x: tokenize(x))
    else:
        df['Token'] = df['contents'].astype(str).apply(lambda x: tokenize(x))
    print('Tokenization time: ', time.time() - start)

    # Save
    print(df)
    df = df.drop(['Function'], axis=1)
    df.to_csv(f'dq_analysis/datasets/{dataset}/tokens.csv', index=False)


def get_df_timestamps(dataset, vul_only=False):
    """ Return the timestamps for a given dataset. """
    # Load vulnerable entries
    if dataset != 'Juliet':
        df = Data(dataset=dataset).get_metadata()
    else:
        return -1

    if vul_only:
        df = df[df['Vulnerable'] == 1]

    # Get timestamps
    df = df[df['Date'] != '-']
    df['Date'] = df['Date'].astype("datetime64")
    df = df[['UID', 'Vulnerable', 'Date']]
    df = df.dropna()
    return df


def compare_token_distribution(dataset):
    """
    Compare the token distribution of old data in comparison to new data.
    """
    # Read the data
    tokens = Data(dataset).get_tokens()
    # Append timestamps
    timestamps = get_df_timestamps(dataset, vul_only=False)
    tokens = tokens.merge(timestamps, on=['UID', 'Vulnerable'], how='inner')
    # Sort entries by time
    tokens = tokens.sort_values(by=['Date'])
    # Split entries by time
    old, new = train_test_split(tokens, test_size=0.5, shuffle=False)

    # Read vocabulary of each
    def read_vocab(token_list):
        vocab = {}
        for tokens in token_list:
            for tok in tokens:
                # Ignore CONSTANTS and STRINGS
                if tok[1] == 'CONSTANT' or tok[1] == 'STRING':
                    continue
                # Record
                if tok[0] not in vocab:
                    vocab[tok[0]] = 0
                vocab[tok[0]] += 1
        return vocab
    old_vocab = read_vocab(old.Token.tolist())
    new_vocab = read_vocab(new.Token.tolist())

    # Normalize the vocabulary lists
    old_freq, new_freq = [], []
    for token in old_vocab:
        # Ensure token is recorded in both lists
        if token not in new_vocab:
            new_vocab[token] = 0

        old_freq.append(old_vocab[token])
        new_freq.append(new_vocab[token])

    # Calculate statistical distance of vectors
    # print('Jensen-Shannon Divergence for', dataset)
    print(f'{dataset} Currency: {jensenshannon(old_freq, new_freq)}')
    print()


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        tokenize_dataset(sys.argv[2])
    elif sys.argv[1] == 'measure':
        compare_token_distribution(sys.argv[2])
    else:
        print(f"ERROR: Unknown command line argument: \"{sys.argv[1]}\"")
