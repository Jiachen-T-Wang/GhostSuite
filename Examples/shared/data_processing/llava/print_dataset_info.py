# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os, sys
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, load_from_disk # huggingface datasets
import argparse

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Filter dataset by topic')
    parser.add_argument('--domain_name', type=str, required=True, help='The topic to filter by')
    args = parser.parse_args()

    finetune_traindata_dir = '/scratch/gpfs/tw8948/LESS/pile-6m/train-{}.bin'.format(args.domain_name)
    finetune_valdata_dir = '/scratch/gpfs/tw8948/LESS/pile-6m/validation-{}.bin'.format(args.domain_name)
    finetune_testdata_dir = '/scratch/gpfs/tw8948/LESS/pile-6m/test-{}.bin'.format(args.domain_name)

    finetune_traindata_REST_dir = '/scratch/gpfs/tw8948/LESS/pile-6m/train-{}-REST.bin'.format(args.domain_name)
    finetune_valdata_REST_dir = '/scratch/gpfs/tw8948/LESS/pile-6m/validation-{}-REST.bin'.format(args.domain_name)
    finetune_testdata_REST_dir = '/scratch/gpfs/tw8948/LESS/pile-6m/test-{}-REST.bin'.format(args.domain_name)

    data = np.memmap(finetune_traindata_dir, dtype=np.uint16, mode='r')
    print(len(data))
    data = np.memmap(finetune_valdata_dir, dtype=np.uint16, mode='r')
    print(len(data))
    data = np.memmap(finetune_testdata_dir, dtype=np.uint16, mode='r')
    print(len(data))

    data = np.memmap(finetune_traindata_REST_dir, dtype=np.uint16, mode='r')
    print(len(data))
    data = np.memmap(finetune_valdata_REST_dir, dtype=np.uint16, mode='r')
    print(len(data))
    data = np.memmap(finetune_testdata_REST_dir, dtype=np.uint16, mode='r')
    print(len(data))