"""
Script to merge two COCO datasets (synthetic chessboards and shuffled chessboards).
Combines images and annotations while ensuring unique IDs.
"""

import json
import os
import shutil
from typing import Dict, List

def load_coco_json(json_path: str) -> Dict:
    """Load COCO format JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def save_coco_json(data: Dict, json_path: str):
    """Save COCO format JSON file."""
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

def merge_coco_datasets(dataset1_path: str, dataset2_path: str, 
                       images_dir1: str, images_dir2: str,
                       output_dir: str, output_json_name: str = 'instances_merged.json'):
    """
    Merge two COCO datasets.
    
    Args:
        dataset1_path: Path to first COCO JSON file
        dataset2_path: Path to second COCO JSON file
        images_dir1: Directory containing images for dataset1
        images_dir2: Directory containing images for dataset2
        output_dir: Output directory for merged dataset
        output_json_name: Name of output JSON file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load both datasets
    print(f"Loading dataset 1: {dataset1_path}")
    dataset1 = load_coco_json(dataset1_path)
    
    print(f"Loading dataset 2: {dataset2_path}")
    dataset2 = load_coco_json(dataset2_path)
    
    # Verify categories match
    if dataset1['categories'] != dataset2['categories']:
        print("Warning: Categories differ between datasets. Using categories from dataset 1.")
    
    # Find maximum IDs to ensure uniqueness
    max_image_id1 = max([img['id'] for img in dataset1['images']], default=0)
    max_ann_id1 = max([ann['id'] for ann in dataset1['annotations']], default=0)
    
    max_image_id2 = max([img['id'] for img in dataset2['images']], default=0)
    max_ann_id2 = max([ann['id'] for ann in dataset2['annotations']], default=0)
    
    # Start new IDs after the maximum of both datasets
    next_image_id = max(max_image_id1, max_image_id2) + 1
    next_ann_id = max(max_ann_id1, max_ann_id2) + 1
    
    # Create mapping for dataset2 IDs
    image_id_mapping = {}  # old_id -> new_id
    new_images = []
    new_annotations = []
    
    # Process dataset2 images and copy them
    print(f"\nProcessing dataset 2 images...")
    for img_info in dataset2['images']:
        old_id = img_info['id']
        new_id = next_image_id
        image_id_mapping[old_id] = new_id
        next_image_id += 1
        
        # Copy image file
        source_path = os.path.join(images_dir2, img_info['file_name'])
        if os.path.exists(source_path):
            dest_path = os.path.join(output_dir, img_info['file_name'])
            if not os.path.exists(dest_path):
                shutil.copy2(source_path, dest_path)
                print(f"  Copied: {img_info['file_name']}")
        else:
            # Try alternative path (maybe it's already in output_dir)
            alt_path = os.path.join(images_dir2, '..', 'synthetic_output', img_info['file_name'])
            if os.path.exists(alt_path):
                dest_path = os.path.join(output_dir, img_info['file_name'])
                if not os.path.exists(dest_path):
                    shutil.copy2(alt_path, dest_path)
                    print(f"  Copied: {img_info['file_name']}")
        
        # Create new image info with new ID
        new_img_info = img_info.copy()
        new_img_info['id'] = new_id
        new_images.append(new_img_info)
    
    # Process dataset2 annotations
    print(f"\nProcessing dataset 2 annotations...")
    for ann in dataset2['annotations']:
        old_image_id = ann['image_id']
        if old_image_id in image_id_mapping:
            new_ann = ann.copy()
            new_ann['id'] = next_ann_id
            new_ann['image_id'] = image_id_mapping[old_image_id]
            next_ann_id += 1
            new_annotations.append(new_ann)
    
    # Copy dataset1 images (if not already present)
    print(f"\nProcessing dataset 1 images...")
    for img_info in dataset1['images']:
        source_path = os.path.join(images_dir1, img_info['file_name'])
        if os.path.exists(source_path):
            dest_path = os.path.join(output_dir, img_info['file_name'])
            if not os.path.exists(dest_path):
                shutil.copy2(source_path, dest_path)
                print(f"  Copied: {img_info['file_name']}")
        else:
            # Try alternative path
            alt_path = os.path.join(images_dir1, 'synthetic_output', img_info['file_name'])
            if os.path.exists(alt_path):
                dest_path = os.path.join(output_dir, img_info['file_name'])
                if not os.path.exists(dest_path):
                    shutil.copy2(alt_path, dest_path)
                    print(f"  Copied: {img_info['file_name']}")
    
    # Create merged dataset
    merged_data = {
        'licenses': dataset1['licenses'],
        'info': {
            **dataset1['info'],
            'description': f"Merged dataset: {dataset1['info'].get('description', '')} + {dataset2['info'].get('description', '')}"
        },
        'categories': dataset1['categories'],
        'images': dataset1['images'] + new_images,
        'annotations': dataset1['annotations'] + new_annotations
    }
    
    # Save merged JSON
    output_json_path = os.path.join(output_dir, output_json_name)
    print(f"\nSaving merged COCO JSON to: {output_json_path}")
    save_coco_json(merged_data, output_json_path)
    
    print(f"\nMerge complete!")
    print(f"  Dataset 1: {len(dataset1['images'])} images, {len(dataset1['annotations'])} annotations")
    print(f"  Dataset 2: {len(dataset2['images'])} images, {len(dataset2['annotations'])} annotations")
    print(f"  Merged: {len(merged_data['images'])} images, {len(merged_data['annotations'])} annotations")

def main():
    # Paths
    synthetic_json = r"C:\MLWorkspace\project\data\synthetic_data\synthetic_output\instances_synthetic.json"
    shuffled_json = r"C:\MLWorkspace\project\data\synthetic_data\shuffled_output\instances_shuffled.json"
    
    synthetic_images_dir = r"C:\MLWorkspace\project\data\synthetic_data\synthetic_output"
    shuffled_images_dir = r"C:\MLWorkspace\project\data\synthetic_data\shuffled_output"
    
    output_dir = r"C:\MLWorkspace\project\data\synthetic_data\merged_output"
    
    # Check if files exist
    if not os.path.exists(synthetic_json):
        print(f"Error: Synthetic dataset not found: {synthetic_json}")
        return
    
    if not os.path.exists(shuffled_json):
        print(f"Error: Shuffled dataset not found: {shuffled_json}")
        return
    
    # Merge datasets
    merge_coco_datasets(
        synthetic_json, shuffled_json,
        synthetic_images_dir, shuffled_images_dir,
        output_dir,
        'instances_merged.json'
    )

if __name__ == '__main__':
    main()

