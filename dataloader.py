import sys
import numpy as np
import torch

from domain_list import PILE_DOMAIN_LIST
from config_file import PILE_DATA_DIR


def load_all_data():

    mixed_train_data = []
    mixed_val_data = []
    mixed_test_data = []

    for domain in PILE_DOMAIN_LIST:

        dom_amp = domain.replace('/', '-').replace(' ', '_').replace('(', '').replace(')', '')

        # Define file paths
        traindata_dir = f'{PILE_DATA_DIR}/train-{dom_amp}.bin'
        valdata_dir = f'{PILE_DATA_DIR}/val/validation-{domain}.bin'
        testdata_dir = f'{PILE_DATA_DIR}/test/test-{domain}.bin'
        
        # Load data
        train_data = np.memmap(traindata_dir, dtype=np.uint16, mode='r')
        val_data = np.memmap(valdata_dir, dtype=np.uint16, mode='r')
        test_data = np.memmap(testdata_dir, dtype=np.uint16, mode='r')

        # Append to mixed data lists
        mixed_train_data.append(train_data)
        mixed_val_data.append(val_data)
        mixed_test_data.append(test_data)
    
    # Concatenate all samples to form the final mixed datasets
    mixed_train_data = np.concatenate(mixed_train_data)
    mixed_val_data = np.concatenate(mixed_val_data)
    mixed_test_data = np.concatenate(mixed_test_data)

    dataset = {'train': mixed_train_data, 
               'val': mixed_val_data, 
               'test': mixed_test_data}
    
    return dataset


def get_batch_from_dataset(split, batch_size, dataset,
              block_size=1024, device='cuda', device_type='cuda',
              i_iter=-1, order_lst=None, return_idx=False, return_first=False,
              generator=None):

    data = dataset[split]
    
    if len(data) - block_size == 0:
        ix = [0]
    elif return_first:
        ix = [0]
    elif order_lst is not None:
        ix = order_lst[i_iter*batch_size:(i_iter+1)*batch_size]
    else:
        if generator is None:
            ix = torch.randint(len(data) - block_size, (batch_size,))
        else:
            ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)

    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    if return_idx:
        return x, y, ix
    else:
        return x, y
