import sys
import json
import numpy as np
import torch

from domain_list import PILE_DOMAIN_LIST

from config_file import DATA_ROOT


class DataSizeError(Exception):
    pass


def load_all_data():

    mixed_train_data = []
    mixed_val_data = []
    mixed_test_data = []

    for domain in PILE_DOMAIN_LIST:

        # Define file paths
        finetune_traindata_dir = f'{DATA_ROOT}/train-{domain}.bin'
        finetune_valdata_dir = f'{DATA_ROOT}/validation-{domain}.bin'
        finetune_testdata_dir = f'{DATA_ROOT}/test-{domain}.bin'
        
        # Load data
        train_data = np.memmap(finetune_traindata_dir, dtype=np.uint16, mode='r')
        val_data = np.memmap(finetune_valdata_dir, dtype=np.uint16, mode='r')
        test_data = np.memmap(finetune_testdata_dir, dtype=np.uint16, mode='r')

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



def load_data_from_domain_mix(domain_proportions, domain_of_interest):
    # Calculate total size of data from domain_of_interest for train, val, and test
    train_data_of_interest = np.memmap(f'{DATA_ROOT}/train-{domain_of_interest}.bin', dtype=np.uint16, mode='r')
    val_data_of_interest = np.memmap(f'{DATA_ROOT}/validation-{domain_of_interest}.bin', dtype=np.uint16, mode='r')
    test_data_of_interest = np.memmap(f'{DATA_ROOT}/test-{domain_of_interest}.bin', dtype=np.uint16, mode='r')

    total_size_of_interest_train = len(train_data_of_interest)
    total_size_of_interest_val = len(val_data_of_interest)
    total_size_of_interest_test = len(test_data_of_interest)

    # Determine the size of data from each domain based on the proportions
    num_samples_train = total_size_of_interest_train / domain_proportions[domain_of_interest]
    num_samples_val = total_size_of_interest_val / domain_proportions[domain_of_interest]
    num_samples_test = total_size_of_interest_test / domain_proportions[domain_of_interest]

    mixed_train_data = []
    mixed_val_data = []
    mixed_test_data = []

    for domain, proportion in domain_proportions.items():
        # Define file paths
        finetune_traindata_dir = f'{DATA_ROOT}/train-{domain}.bin'
        finetune_valdata_dir = f'{DATA_ROOT}/validation-{domain}.bin'
        finetune_testdata_dir = f'{DATA_ROOT}/test-{domain}.bin'
        
        # Load data
        train_data = np.memmap(finetune_traindata_dir, dtype=np.uint16, mode='r')
        val_data = np.memmap(finetune_valdata_dir, dtype=np.uint16, mode='r')
        test_data = np.memmap(finetune_testdata_dir, dtype=np.uint16, mode='r')

        # Calculate the number of samples to take from each domain
        samples_train = int(num_samples_train * proportion)
        samples_val = int(num_samples_val * proportion)
        samples_test = int(num_samples_test * proportion)

        sampled_train_data = train_data[:samples_train]
        sampled_val_data = val_data[:samples_val]
        sampled_test_data = test_data[:samples_test]

        # Append to mixed data lists
        mixed_train_data.append(sampled_train_data)
        mixed_val_data.append(sampled_val_data)
        mixed_test_data.append(sampled_test_data)
    
    # Concatenate all samples to form the final mixed datasets
    mixed_train_data = np.concatenate(mixed_train_data)
    mixed_val_data = np.concatenate(mixed_val_data)
    mixed_test_data = np.concatenate(mixed_test_data)

    dataset = {'train': mixed_train_data, 
               'val': mixed_val_data, 
               'test': mixed_test_data}
    
    return dataset


def extract_key_value(file_path, key, s=1000):

  # Open the text file containing the data
  with open(file_path, 'r') as file:
      # Read all lines from the file
      lines = file.readlines()

  # Initialize an empty list to store test_loss values
  test_loss_values = []

  key = '"'+key+'":'

  count = 0

  # Iterate through each line
  for line in lines:
      # Check if 'test_loss' is in the line
      if key in line:
          count += 1
          # Find the start index of the numeric value following 'test_loss':
          start_index = line.find(key) + len(key)
          # Extract the substring containing the number
          numeric_part = line[start_index:].split(',')[0].strip()
          # Convert to float and append to the list
          test_loss_values.append(float(numeric_part))
          if count > s:
            break

  # Now you have a list of all test_loss values
  return test_loss_values
