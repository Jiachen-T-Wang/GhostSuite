import os
import numpy as np
import torch

# Local imports
from .domain_list import PILE_DOMAIN_LIST

PILE_DATA_DIR_TRAIN = '/scratch/gpfs/PMITTAL/tianhao/PretrainData/pile/pile-train'
PILE_DATA_DIR_VAL = '/scratch/gpfs/PMITTAL/tianhao/PretrainData/pile/pile-val-gpt2'
PILE_DATA_DIR_TEST = '/scratch/gpfs/PMITTAL/tianhao/PretrainData/pile/pile-test-gpt2'


def _get_domain_paths(domain_list):
    domain_paths = []
    for domain in domain_list:
        dom_amp = domain.replace('/', '-').replace(' ', '_').replace('(', '').replace(')', '')
        domain_paths.append(
            {
                'name': domain,
                'train': f'{PILE_DATA_DIR_TRAIN}/train-{dom_amp}.bin',
                'val': f'{PILE_DATA_DIR_VAL}/validation-{dom_amp}.bin',
                'test': f'{PILE_DATA_DIR_TEST}/test-{dom_amp}.bin'
            }
        )
    return domain_paths


def _allocate_train_tokens(domain_paths, token_budget):
    bytes_per_token = np.dtype(np.uint16).itemsize
    train_token_counts = {
        domain['name']: os.path.getsize(domain['train']) // bytes_per_token for domain in domain_paths
    }
    total_tokens = sum(train_token_counts.values())
    if total_tokens == 0:
        raise ValueError('No training tokens found for the requested domains.')

    effective_budget = min(token_budget, total_tokens)
    allocations = {}
    remainders = []
    for domain, tokens in train_token_counts.items():
        exact = (tokens / total_tokens) * effective_budget
        base = min(int(exact), tokens)
        allocations[domain] = base
        remainders.append((domain, exact - base))

    used = sum(allocations.values())
    leftover = effective_budget - used
    if leftover > 0:
        for domain, _ in sorted(remainders, key=lambda item: item[1], reverse=True):
            if leftover == 0:
                break
            if allocations[domain] < train_token_counts[domain]:
                allocations[domain] += 1
                leftover -= 1

    if sum(allocations.values()) == 0:
        raise ValueError('token_budget is too small to allocate any tokens.')

    return allocations


def load_all_data(token_budget: int=1_000_000_000):

    mixed_train_data = []
    mixed_val_data = []
    mixed_test_data = []

    # Get the list of domains to load
    domain_list = PILE_DOMAIN_LIST
    domain_paths = _get_domain_paths(domain_list)

    train_allocations = None
    if token_budget is not None:
        if token_budget <= 0:
            raise ValueError('token_budget must be positive when provided.')
        train_allocations = _allocate_train_tokens(domain_paths, token_budget)

    for domain in domain_paths:
        train_shape = None
        if train_allocations is not None:
            allocated_tokens = train_allocations.get(domain['name'], 0)
            if allocated_tokens == 0:
                continue
            train_shape = (allocated_tokens,)

        train_data = np.memmap(domain['train'], dtype=np.uint16, mode='r', shape=train_shape)
        mixed_train_data.append(train_data)

    for domain in domain_paths:
        val_data = np.memmap(domain['val'], dtype=np.uint16, mode='r')
        mixed_val_data.append(val_data)

    for domain in domain_paths:
        test_data = np.memmap(domain['test'], dtype=np.uint16, mode='r')
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
