#!/usr/bin/env python3
"""
Tokenize Pile dataset by domain with streaming processing and global token limit.
"""

import os
import numpy as np
from datasets import load_dataset
from transformers import GPT2Tokenizer
from collections import defaultdict
from tqdm import tqdm
import argparse

class DomainFileManager:
    """Manages binary files for each domain with append functionality."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.file_handles = {}
        self.token_counts = defaultdict(int)
        
    def get_filepath(self, split, domain):
        """Generate clean filepath for domain."""
        clean_domain = domain.replace('/', '-').replace(' ', '_').replace('(', '').replace(')', '')
        filename = f"{split}-{clean_domain}.bin"
        return os.path.join(self.output_dir, filename)
    
    def append_tokens(self, split, domain, tokens):
        """Append tokens to domain file."""
        filepath = self.get_filepath(split, domain)
        
        # Convert to numpy array
        token_array = np.array(tokens, dtype=np.uint16)
        
        # Append to file
        with open(filepath, 'ab') as f:  # 'ab' for append binary
            token_array.tofile(f)
        
        # Update token count
        self.token_counts[f"{split}-{domain}"] += len(tokens)
        
        return len(tokens)
    
    def get_total_tokens(self):
        """Get total tokens processed across all domains."""
        return sum(self.token_counts.values())
    
    def cleanup_empty_files(self):
            """Remove any empty files that might have been created."""
            # Correctly iterate through files in the output directory
            for filename in os.listdir(self.output_dir):
                if filename.endswith('.bin'):
                    filepath = os.path.join(self.output_dir, filename)
                    # Check if the file exists and is empty
                    if os.path.exists(filepath) and os.path.getsize(filepath) == 0:
                        print(f"Removing empty file: {filename}")
                        os.remove(filepath)
    
    def print_summary(self):
        """Print summary of saved files."""
        print("\nSaved files:")
        all_files = {}
        
        for filename in sorted(os.listdir(self.output_dir)):
            if filename.endswith('.bin'):
                filepath = os.path.join(self.output_dir, filename)
                if os.path.exists(filepath):
                    file_size = os.path.getsize(filepath)
                    num_tokens = file_size // 2  # uint16 = 2 bytes
                    all_files[filename] = (num_tokens, file_size)
        
        for filename, (num_tokens, file_size) in all_files.items():
            print(f"  {filename}: {num_tokens:,} tokens ({file_size / 1024 / 1024:.1f} MB)")

def main():
    parser = argparse.ArgumentParser(description='Tokenize Pile dataset by domain with streaming')
    parser.add_argument('--output_dir', type=str, default='/scratch/gpfs/tw8948/pile_tokenized', 
                       help='Directory to save tokenized files')
    parser.add_argument('--num_proc', type=int, default=8, 
                       help='Number of processes for dataset loading')
    parser.add_argument('--batch_size', type=int, default=100, 
                       help='Batch size for processing')
    parser.add_argument('--max_length', type=int, default=None,
                       help='Maximum sequence length (None for no limit)')
    parser.add_argument('--max_tokens', type=int, default=30_000_000_000,
                       help='Maximum total tokens to process (default: 15B)')
    parser.add_argument('--progress_interval', type=int, default=10000,
                       help='Progress update interval')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize file manager
    file_manager = DomainFileManager(args.output_dir)
    
    # Load dataset with streaming to avoid memory issues
    print("Loading Pile dataset...")
    dataset_name = 'monology/pile-uncopyrighted'
    try:
        dataset = load_dataset(dataset_name, num_proc=args.num_proc, download_mode="reuse_dataset_if_exists")
    except TypeError as e:
        if "promote_options" in str(e):
            print("PyArrow version incompatibility detected. Using streaming dataset...")
            dataset = load_dataset(dataset_name, streaming=True)
        else:
            raise e
    
    # Initialize GPT-2 tokenizer
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    total_processed = 0
    total_examples = 0
    
    # Process each split
    for split in dataset.keys():
        print(f"\nProcessing {split} split...")
        
        # Handle both regular and streaming datasets
        split_data = dataset[split]
        
        # For streaming datasets, we can't get total length
        if hasattr(split_data, '__len__'):
            pbar = tqdm(total=len(split_data), desc=f"Processing {split}")
        else:
            pbar = tqdm(desc=f"Processing {split} (streaming)")
        
        # Process examples in batches
        batch_examples = []
        
        for example in split_data:
            batch_examples.append(example)
            total_examples += 1
            
            # Process batch when it reaches batch_size
            if len(batch_examples) >= args.batch_size:
                tokens_added = process_batch(batch_examples, split, tokenizer, file_manager, args)
                total_processed += tokens_added
                batch_examples = []
                
                pbar.update(args.batch_size)
                
                # Update progress description
                if total_examples % args.progress_interval == 0:
                    pbar.set_description(
                        f"Processing {split} - {total_processed/1e9:.2f}B tokens"
                    )
                
                # Check if we've hit the token limit
                if total_processed >= args.max_tokens:
                    print(f"\nReached token limit of {args.max_tokens:,} tokens. Stopping.")
                    pbar.close()
                    break
            
            # Early exit if token limit reached
            if total_processed >= args.max_tokens:
                break
        
        # Process remaining examples in the batch
        if batch_examples and total_processed < args.max_tokens:
            tokens_added = process_batch(batch_examples, split, tokenizer, file_manager, args)
            total_processed += tokens_added
            pbar.update(len(batch_examples))
        
        pbar.close()
        
        # Exit if we've reached the token limit
        if total_processed >= args.max_tokens:
            break
    
    print(f"\nTokenization complete!")
    print(f"Total tokens processed: {total_processed:,} ({total_processed/1e9:.2f}B)")
    print(f"Total examples processed: {total_examples:,}")
    print(f"Files saved in: {args.output_dir}")
    
    # Clean up and print summary
    file_manager.cleanup_empty_files()
    file_manager.print_summary()

def process_batch(batch_examples, split, tokenizer, file_manager, args):
    """Process a batch of examples and append tokens to appropriate domain files."""
    tokens_added = 0
    
    # Group batch by domain
    domain_batches = defaultdict(list)
    for example in batch_examples:
        domain = example['meta']['pile_set_name']
        text = example['text']
        domain_batches[domain].append(text)
    
    # Process each domain in the batch
    for domain, texts in domain_batches.items():
        # Tokenize texts for this domain
        if args.max_length:
            batch_tokens = tokenizer(
                texts, 
                truncation=True, 
                max_length=args.max_length,
                return_attention_mask=False,
                return_token_type_ids=False
            )['input_ids']
        else:
            batch_tokens = tokenizer(
                texts,
                return_attention_mask=False,
                return_token_type_ids=False
            )['input_ids']
        
        # Flatten tokens and add EOS tokens between documents
        domain_tokens = []
        for tokens in batch_tokens:
            domain_tokens.extend(tokens)
            domain_tokens.append(tokenizer.eos_token_id)
        
        # Append to domain file
        if domain_tokens:
            added = file_manager.append_tokens(split, domain, domain_tokens)
            tokens_added += added
    
    return tokens_added

def test_loading(output_dir='./tokenized_pile'):
    """Test function to verify the saved files can be loaded correctly."""
    print("\nTesting file loading...")
    
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist.")
        return
    
    for filename in os.listdir(output_dir):
        if filename.endswith('.bin'):
            filepath = os.path.join(output_dir, filename)
            
            # Load with memmap
            domain_data = np.memmap(filepath, dtype=np.uint16, mode="r")
            print(f"{filename}: {len(domain_data)} tokens, first 10: {domain_data[:10]}")
            
            # Test a small slice
            sample = domain_data[:100] if len(domain_data) >= 100 else domain_data[:]
            print(f"  Sample shape: {sample.shape}, dtype: {sample.dtype}")

if __name__ == "__main__":
    main()
    
    # Uncomment to test loading after tokenization
    # test_loading()


