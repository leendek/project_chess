"""
Script to create synthetic chessboard images by copying rows.
Copies row 1 (top row) to all odd rows (1, 3, 5, 7)
Copies row 8 (bottom row) to all even rows (2, 4, 6, 8)
"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple

def load_coco_json(json_path: str) -> Dict:
    """Load COCO format JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def save_coco_json(data: Dict, json_path: str):
    """Save COCO format JSON file."""
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

def get_board_bbox(annotations: List[Dict], image_id: int, category_id: int = 1) -> Tuple[float, float, float, float]:
    """
    Get the board bounding box for an image.
    Returns (x, y, width, height) or None if not found.
    """
    for ann in annotations:
        if ann['image_id'] == image_id and ann['category_id'] == category_id:
            bbox = ann['bbox']  # [x, y, width, height]
            return bbox[0], bbox[1], bbox[2], bbox[3]
    return None

def get_row_for_annotation(ann: Dict, board_bbox: Tuple[float, float, float, float], image_height: int) -> int:
    """
    Determine which row (0-7) an annotation belongs to.
    Row 0 is top row, row 7 is bottom row.
    """
    x, y, w, h = board_bbox
    board_y1 = y
    board_y2 = y + h
    
    # Get center y of annotation
    ann_bbox = ann['bbox']  # [x, y, width, height]
    ann_center_y = ann_bbox[1] + ann_bbox[3] / 2
    
    # Map to row (0-7)
    relative_y = (ann_center_y - board_y1) / (board_y2 - board_y1)
    row = int(relative_y * 8)
    row = max(0, min(7, row))  # Clamp to 0-7
    
    return row

def copy_row_in_image(image: np.ndarray, board_bbox: Tuple[float, float, float, float], 
                      source_row: int, target_row: int) -> np.ndarray:
    """
    Copy a row from source_row to target_row in the image.
    Returns modified image array.
    """
    x, y, w, h = board_bbox
    board_y1 = int(y)
    board_y2 = int(y + h)
    board_x1 = int(x)
    board_x2 = int(x + w)
    
    # Calculate row boundaries consistently
    # Use integer division to ensure consistent row heights
    board_height = board_y2 - board_y1
    row_height = board_height // 8  # Integer division for consistent height
    
    # Calculate source row bounds
    source_y1 = board_y1 + source_row * row_height
    source_y2 = source_y1 + row_height
    
    # Calculate target row bounds - use same height
    target_y1 = board_y1 + target_row * row_height
    target_y2 = target_y1 + row_height
    
    # Ensure we don't go out of bounds
    source_y2 = min(source_y2, board_y2)
    target_y2 = min(target_y2, board_y2)
    
    # Get the actual heights
    source_height = source_y2 - source_y1
    target_height = target_y2 - target_y1
    
    # Copy the row - use the minimum height to avoid size mismatch
    copy_height = min(source_height, target_height)
    
    if copy_height > 0:
        image[target_y1:target_y1+copy_height, board_x1:board_x2] = \
            image[source_y1:source_y1+copy_height, board_x1:board_x2].copy()
    
    return image

def adjust_annotation_bbox(ann: Dict, source_row: int, target_row: int, 
                          board_bbox: Tuple[float, float, float, float]) -> Dict:
    """
    Adjust annotation bbox to move from source_row to target_row.
    Returns new annotation dict.
    """
    x, y, w, h = board_bbox
    row_height = h / 8
    
    # Calculate y offset
    source_y_center = y + (source_row + 0.5) * row_height
    target_y_center = y + (target_row + 0.5) * row_height
    y_offset = target_y_center - source_y_center
    
    # Create new annotation
    new_ann = ann.copy()
    new_bbox = ann['bbox'].copy()
    new_bbox[1] += y_offset  # Adjust y coordinate
    new_ann['bbox'] = new_bbox
    
    return new_ann

def process_image(coco_data: Dict, image_info: Dict, images_dir: str, output_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Process a single image to create synthetic version.
    Returns (new_image_info, new_annotations).
    """
    image_id = image_info['id']
    image_filename = image_info['file_name']
    image_width = image_info['width']
    image_height = image_info['height']
    
    # Load image
    image_path = os.path.join(images_dir, image_filename)
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return None, []
    
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    
    # Get board bounding box
    board_bbox = get_board_bbox(coco_data['annotations'], image_id)
    if board_bbox is None:
        print(f"Warning: No board found for image {image_id}")
        return None, []
    
    # Get annotations for this image
    image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    
    # Separate board annotation from piece annotations
    board_ann = None
    piece_annotations = []
    for ann in image_annotations:
        if ann['category_id'] == 1:  # board category
            board_ann = ann
        else:
            piece_annotations.append(ann)
    
    # Identify rows for each piece annotation
    row_annotations = {i: [] for i in range(8)}
    for ann in piece_annotations:
        row = get_row_for_annotation(ann, board_bbox, image_height)
        row_annotations[row].append(ann)
    
    # Copy rows in image
    # Copy row 0 (first row) to rows 0, 2, 4, 6 (odd rows: 1, 3, 5, 7)
    for target_row in [0, 2, 4, 6]:
        if target_row != 0:  # Don't copy row 0 to itself
            image_array = copy_row_in_image(image_array, board_bbox, 0, target_row)
    
    # Copy row 7 (last row) to rows 1, 3, 5, 7 (even rows: 2, 4, 6, 8)
    for target_row in [1, 3, 5, 7]:
        if target_row != 7:  # Don't copy row 7 to itself
            image_array = copy_row_in_image(image_array, board_bbox, 7, target_row)
    
    # Create new annotations
    new_annotations = []
    
    # Keep board annotation as is
    if board_ann:
        new_annotations.append(board_ann.copy())
    
    # Copy piece annotations
    # Copy row 0 pieces to rows 0, 2, 4, 6 (chess rows 1, 3, 5, 7)
    for source_row in [0]:
        for target_row in [0, 2, 4, 6]:
            for ann in row_annotations[source_row]:
                if target_row == source_row:
                    # Keep original annotation for row 0
                    new_ann = ann.copy()
                else:
                    # Copy to other odd rows
                    new_ann = adjust_annotation_bbox(ann, source_row, target_row, board_bbox)
                new_annotations.append(new_ann)
    
    # Copy row 7 pieces to rows 1, 3, 5, 7 (chess rows 2, 4, 6, 8)
    for source_row in [7]:
        for target_row in [1, 3, 5, 7]:
            for ann in row_annotations[source_row]:
                if target_row == source_row:
                    # Keep original annotation for row 7
                    new_ann = ann.copy()
                else:
                    # Copy to other even rows
                    new_ann = adjust_annotation_bbox(ann, source_row, target_row, board_bbox)
                new_annotations.append(new_ann)
    
    # Save new image
    new_image_filename = f"synthetic_{image_filename}"
    new_image_path = os.path.join(output_dir, new_image_filename)
    new_image = Image.fromarray(image_array)
    new_image.save(new_image_path)
    
    # Create new image info
    new_image_info = image_info.copy()
    new_image_info['file_name'] = new_image_filename
    
    return new_image_info, new_annotations

def main():
    # Paths
    coco_json_path = r"C:\MLWorkspace\project\data\synthetic_data\instances_default.json"
    images_dir = r"C:\MLWorkspace\project\data\synthetic_data"
    output_dir = r"C:\MLWorkspace\project\data\synthetic_data\synthetic_output"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load COCO data
    print(f"Loading COCO JSON from: {coco_json_path}")
    coco_data = load_coco_json(coco_json_path)
    
    # Copy original images to output directory
    print("Copying original images to output directory...")
    for image_info in coco_data['images']:
        original_image_path = os.path.join(images_dir, image_info['file_name'])
        if os.path.exists(original_image_path):
            output_image_path = os.path.join(output_dir, image_info['file_name'])
            if not os.path.exists(output_image_path):
                shutil.copy2(original_image_path, output_image_path)
                print(f"  Copied: {image_info['file_name']}")
    
    # Process each image to create synthetic versions
    new_images = []
    new_annotations = []
    next_image_id = max([img['id'] for img in coco_data['images']], default=0) + 1
    next_ann_id = max([ann['id'] for ann in coco_data['annotations']], default=0) + 1
    
    for image_info in coco_data['images']:
        print(f"Processing image: {image_info['file_name']}")
        result = process_image(coco_data, image_info, images_dir, output_dir)
        
        if result[0] is None:
            continue
        
        new_image_info, new_image_annotations = result
        
        # Assign new IDs
        new_image_info['id'] = next_image_id
        next_image_id += 1
        
        for ann in new_image_annotations:
            ann['id'] = next_ann_id
            ann['image_id'] = new_image_info['id']
            next_ann_id += 1
        
        new_images.append(new_image_info)
        new_annotations.extend(new_image_annotations)
    
    # Create new COCO dataset
    new_coco_data = {
        'licenses': coco_data['licenses'],
        'info': coco_data['info'],
        'categories': coco_data['categories'],
        'images': coco_data['images'] + new_images,
        'annotations': coco_data['annotations'] + new_annotations
    }
    
    # Save new COCO JSON
    output_json_path = os.path.join(output_dir, 'instances_synthetic.json')
    print(f"Saving synthetic COCO JSON to: {output_json_path}")
    save_coco_json(new_coco_data, output_json_path)
    
    print(f"\nDone! Created {len(new_images)} synthetic images.")
    print(f"Total images: {len(new_coco_data['images'])}")
    print(f"Total annotations: {len(new_coco_data['annotations'])}")

if __name__ == '__main__':
    main()

