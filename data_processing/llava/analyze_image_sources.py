import json
import os
from collections import Counter
import re

def analyze_image_sources():
    dataset_dir = '/scratch/gpfs/tw8948/llava_dataset/'
    json_files = [
        'complex_reasoning_77k.json', 
        'detail_23k.json', 
        'llava_instruct_80k.json',
        'conversation_58k.json',
        'llava_instruct_150k.json'
    ]
    
    all_image_paths = []
    image_prefixes = Counter()
    
    for json_file in json_files:
        json_path = os.path.join(dataset_dir, json_file)
        if not os.path.exists(json_path):
            print(f"File not found: {json_path}")
            continue
            
        print(f"\n=== Analyzing {json_file} ===")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        file_image_paths = []
        for item in data:
            if 'image' in item:
                image_path = item['image']
                file_image_paths.append(image_path)
                all_image_paths.append(image_path)
                
                # Extract prefix/directory to understand source
                if '/' in image_path:
                    prefix = image_path.split('/')[0]
                else:
                    # For files like "000000129772.jpg", likely COCO
                    if re.match(r'^\d{12}\.jpg$', image_path):
                        prefix = 'coco'
                    else:
                        prefix = 'other'
                
                image_prefixes[prefix] += 1
        
        print(f"  Found {len(file_image_paths)} image paths")
        
        # Show some examples
        if file_image_paths:
            print("  Sample paths:")
            for i, path in enumerate(file_image_paths[:5]):
                print(f"    {path}")
    
    print(f"\n=== OVERALL SUMMARY ===")
    print(f"Total images: {len(all_image_paths)}")
    print(f"Image source breakdown:")
    for prefix, count in image_prefixes.most_common():
        percentage = (count / len(all_image_paths)) * 100
        print(f"  {prefix}: {count} images ({percentage:.1f}%)")
    
    # Map prefixes to required downloads
    print(f"\n=== REQUIRED DOWNLOADS ===")
    download_mapping = {
        'coco': 'COCO Train 2017 (http://images.cocodataset.org/zips/train2017.zip)',
        'gqa': 'GQA Images (https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)',
        'ocr_vqa': 'OCR VQA Images (https://huggingface.co/datasets/qnguyen3/ocr_vqa/resolve/main/ocr_vqa.zip)',
        'textvqa': 'TextVQA Images (https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)',
        'vg': 'Visual Genome Images (https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip + images2.zip)',
        'VG_100K': 'Visual Genome Images (https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip)',
        'VG_100K_2': 'Visual Genome Images Part 2 (https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)'
    }
    
    for prefix in image_prefixes:
        if prefix in download_mapping:
            print(f"  ✓ {download_mapping[prefix]}")
        else:
            print(f"  ? Unknown source '{prefix}' - need to investigate")
    
    # Check for numeric-only filenames (likely COCO)
    coco_pattern_count = sum(1 for path in all_image_paths if re.match(r'^\d{12}\.jpg$', path))
    if coco_pattern_count > 0:
        print(f"\n  Note: {coco_pattern_count} images match COCO pattern (############.jpg)")

if __name__ == "__main__":
    analyze_image_sources()