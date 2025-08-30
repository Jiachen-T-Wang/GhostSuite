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
num_proc = 40

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Filter dataset by topic')
    parser.add_argument('--topic', type=str, required=True, help='The topic to filter by')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory of tokenized dataset')
    parser.add_argument('--metainfo', type=str, required=True, help='key name of meta info')    

    parser.add_argument('--valonly', action='store_true', help='Enable validation only mode')
    parser.add_argument('--trainonly', action='store_true', help='Enable train only mode')
    parser.add_argument('--exclude', action='store_true', help='Exclude this topic')

    args = parser.parse_args()

    topic_to_filter = args.topic
    data_dir = args.data_dir
    metainfo = args.metainfo
    # data_dir = "/scratch/gpfs/tw8948/LESS/pile-full"
    
    tokenized = load_from_disk(data_dir)

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():

        print('--------------------------------')
        print('Current Split:', split)
        print('Data Information:')
        print(dset)
        print('--------------------------------')

        if args.valonly:
            if split == 'train':
                continue

        if args.trainonly:
            if split in ['validation', 'test']:
                continue

        if args.exclude:
            dset = dset.filter(lambda example: example['meta'][metainfo] != topic_to_filter)
            print('Take only corpus with topic not in *{}*'.format(topic_to_filter))
            filename = os.path.join(data_dir, '{}-{}-REST.bin'.format(split, topic_to_filter))
        else:
            dset = dset.filter(lambda example: example['meta'][metainfo] == topic_to_filter)
            print('Take only corpus with topic in *{}*'.format(topic_to_filter))
            filename = os.path.join(data_dir, '{}-{}.bin'.format(split, topic_to_filter))

        arr_len = np.sum(dset['len'], dtype=np.uint64)

        print('Filtered dataset')
        print(dset)
        print('Total Token Length for {}: {}'.format(topic_to_filter, arr_len))

        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        if split in ['train']:
            total_batches = 1024
        else:
            total_batches = 12 # Number of shards the dataset is divided into

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, 
                               index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        