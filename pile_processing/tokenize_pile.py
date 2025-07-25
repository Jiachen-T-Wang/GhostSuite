import os, sys
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

# Replace by your own directory. 
TOKENIZED_DATA_DIR = "/scratch/gpfs/tw8948/GhostSuite/pile-tokenized"
DATA_DIR = "/scratch/gpfs/tw8948/GhostSuite/pile-bin"

num_proc = 32
enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':

    dataset_name = 'monology/pile-uncopyrighted'
    dataset = load_dataset( dataset_name, num_proc=num_proc, download_mode="reuse_dataset_if_exists" )
    print('Pile-uncopyrighted dataset loaded')
    print(dataset)

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        out = {'ids': ids, 'len': len(ids), 'meta': example['meta']}
        return out

    # tokenize the dataset
    tokenized = dataset.map(
        process,
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    tokenized.save_to_disk(f"{TOKENIZED_DATA_DIR}")
    print('saved tokenized data!')

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():

        print(split)
        print(dset)

        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(f'{DATA_DIR}', f'{split}.bin')

        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

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