"""
Script to shuffle chessboard pieces while maintaining checkerboard pattern.
Black pieces stay on black squares, white pieces stay on white squares.
"""

import json
import os
import shutil
import random
import argparse
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
    """Get the board bounding box for an image."""
    for ann in annotations:
        if ann['image_id'] == image_id and ann['category_id'] == category_id:
            bbox = ann['bbox']
            return bbox[0], bbox[1], bbox[2], bbox[3]
    return None

def get_square_for_annotation(ann: Dict, board_bbox: Tuple[float, float, float, float]) -> Tuple[int, int]:
    """Determine which square (row, col) an annotation belongs to. Returns (row, col) 0-7."""
    x, y, w, h = board_bbox
    board_x1, board_y1 = x, y
    board_x2, board_y2 = x + w, y + h
    
    ann_bbox = ann['bbox']
    ann_center_x = ann_bbox[0] + ann_bbox[2] / 2
    ann_center_y = ann_bbox[1] + ann_bbox[3] / 2
    
    relative_x = (ann_center_x - board_x1) / (board_x2 - board_x1)
    relative_y = (ann_center_y - board_y1) / (board_y2 - board_y1)
    
    col = int(relative_x * 8)
    row = int(relative_y * 8)
    col = max(0, min(7, col))
    row = max(0, min(7, row))
    
    return row, col

def is_black_square(row: int, col: int) -> bool:
    """Check if a square is black.
    Row 1 (index 0): even squares (0,2,4,6) are black, odd squares (1,3,5,7) are white
    Row 2 (index 1): even squares are white, odd squares are black
    Pattern alternates.
    """
    # Square is black if row and col have the same parity (both even or both odd)
    return (row % 2) == (col % 2)

def copy_square_in_image(image: np.ndarray, board_bbox: Tuple[float, float, float, float],
                         source_row: int, source_col: int, target_row: int, target_col: int,
                         source_image: np.ndarray = None) -> np.ndarray:
    """Copy a square from source to target position.
    If source_image is provided, copy from that instead of the main image.
    """
    x, y, w, h = board_bbox
    board_x1, board_y1 = int(x), int(y)
    board_x2, board_y2 = int(x + w), int(y + h)
    
    board_width = board_x2 - board_x1
    board_height = board_y2 - board_y1
    square_width = board_width // 8
    square_height = board_height // 8
    
    source_x1 = source_col * square_width
    source_y1 = source_row * square_height
    source_x2 = source_x1 + square_width
    source_y2 = source_y1 + square_height
    
    target_x1 = board_x1 + target_col * square_width
    target_y1 = board_y1 + target_row * square_height
    target_x2 = target_x1 + square_width
    target_y2 = target_y1 + square_height
    
    # Ensure bounds
    source_x2 = min(source_x2, board_width)
    source_y2 = min(source_y2, board_height)
    target_x2 = min(target_x2, board_x2)
    target_y2 = min(target_y2, board_y2)
    
    copy_width = min(source_x2 - source_x1, target_x2 - target_x1)
    copy_height = min(source_y2 - source_y1, target_y2 - target_y1)
    
    if copy_width > 0 and copy_height > 0:
        if source_image is not None:
            # Copy from source_image (which is already relative to board)
            image[target_y1:target_y1+copy_height, target_x1:target_x1+copy_width] = \
                source_image[source_y1:source_y1+copy_height, source_x1:source_x1+copy_width].copy()
        else:
            # Copy from main image
            src_x1_abs = board_x1 + source_col * square_width
            src_y1_abs = board_y1 + source_row * square_height
            image[target_y1:target_y1+copy_height, target_x1:target_x1+copy_width] = \
                image[src_y1_abs:src_y1_abs+copy_height, src_x1_abs:src_x1_abs+copy_width].copy()
    
    return image

def adjust_annotation_to_square(ann: Dict, target_row: int, target_col: int,
                                board_bbox: Tuple[float, float, float, float]) -> Dict:
    """Adjust annotation bbox to move to target square."""
    x, y, w, h = board_bbox
    board_x1, board_y1 = x, y
    
    square_width = w / 8
    square_height = h / 8
    
    # Calculate target square center
    target_center_x = board_x1 + (target_col + 0.5) * square_width
    target_center_y = board_y1 + (target_row + 0.5) * square_height
    
    # Get original bbox dimensions
    ann_bbox = ann['bbox']
    ann_width = ann_bbox[2]
    ann_height = ann_bbox[3]
    
    # Create new annotation centered on target square
    new_ann = ann.copy()
    new_ann['bbox'] = [
        target_center_x - ann_width / 2,
        target_center_y - ann_height / 2,
        ann_width,
        ann_height
    ]
    
    return new_ann

def create_shuffled_image(image_array: np.ndarray, board_bbox: Tuple[float, float, float, float],
                          black_square_pieces: List, white_square_pieces: List,
                          board_ann: Dict) -> Tuple[np.ndarray, List[Dict]]:
    """Create a shuffled version of the chessboard image.
    Returns (shuffled_image_array, new_annotations).
    """
    # Get all black and white square positions
    black_squares = [(r, c) for r in range(8) for c in range(8) if is_black_square(r, c)]
    white_squares = [(r, c) for r in range(8) for c in range(8) if not is_black_square(r, c)]
    
    # Shuffle the square positions (different shuffle each time)
    shuffled_black_squares = black_squares.copy()
    shuffled_white_squares = white_squares.copy()
    random.shuffle(shuffled_black_squares)
    random.shuffle(shuffled_white_squares)
    
    # Create mapping from original to shuffled positions
    black_mapping = {orig: shuffled for orig, shuffled in zip(black_squares, shuffled_black_squares)}
    white_mapping = {orig: shuffled for orig, shuffled in zip(white_squares, shuffled_white_squares)}
    
    # Copy squares in image - use temporary copy to avoid overwriting
    x, y, w, h = board_bbox
    board_x1, board_y1 = int(x), int(y)
    board_x2, board_y2 = int(x + w), int(y + h)
    
    # Create temporary copy of board region
    temp_board = image_array[board_y1:board_y2, board_x1:board_x2].copy()
    
    # Create a copy of the image to modify
    shuffled_image = image_array.copy()
    
    # Copy squares from temp to their new positions
    for orig_pos, target_pos in black_mapping.items():
        copy_square_in_image(shuffled_image, board_bbox, 
                            orig_pos[0], orig_pos[1],
                            target_pos[0], target_pos[1],
                            source_image=temp_board)
    
    for orig_pos, target_pos in white_mapping.items():
        copy_square_in_image(shuffled_image, board_bbox,
                            orig_pos[0], orig_pos[1],
                            target_pos[0], target_pos[1],
                            source_image=temp_board)
    
    # Create new annotations
    new_annotations = []
    if board_ann:
        new_annotations.append(board_ann.copy())
    
    # Update piece annotations
    for row, col, ann in black_square_pieces:
        target_row, target_col = black_mapping[(row, col)]
        new_ann = adjust_annotation_to_square(ann, target_row, target_col, board_bbox)
        new_annotations.append(new_ann)
    
    for row, col, ann in white_square_pieces:
        target_row, target_col = white_mapping[(row, col)]
        new_ann = adjust_annotation_to_square(ann, target_row, target_col, board_bbox)
        new_annotations.append(new_ann)
    
    return shuffled_image, new_annotations

def process_image(coco_data: Dict, image_info: Dict, images_dir: str, output_dir: str, 
                  shuffle_index: int = 0) -> Tuple[List[Dict], List[Dict]]:
    """Process image to create shuffled version.
    
    Args:
        coco_data: COCO dataset dictionary
        image_info: Image info dictionary
        images_dir: Directory containing source images
        output_dir: Output directory for shuffled images
        shuffle_index: Index for this shuffle (used in filename)
    
    Returns:
        (new_image_info, new_annotations) or (None, []) if processing fails
    """
    image_id = image_info['id']
    image_filename = image_info['file_name']
    
    image_path = os.path.join(images_dir, image_filename)
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return None, []
    
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    
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
        if ann['category_id'] == 1:  # board
            board_ann = ann
        else:
            piece_annotations.append(ann)
    
    # Group pieces by square color
    black_square_pieces = []  # (row, col, annotation)
    white_square_pieces = []
    
    for ann in piece_annotations:
        row, col = get_square_for_annotation(ann, board_bbox)
        if is_black_square(row, col):
            black_square_pieces.append((row, col, ann))
        else:
            white_square_pieces.append((row, col, ann))
    
    # Create shuffled version
    shuffled_image, new_annotations = create_shuffled_image(
        image_array, board_bbox, black_square_pieces, white_square_pieces, board_ann
    )
    
    # Save new image
    base_name = os.path.splitext(image_filename)[0]
    ext = os.path.splitext(image_filename)[1]
    new_image_filename = f"shuffled_{shuffle_index:03d}_{base_name}{ext}"
    new_image_path = os.path.join(output_dir, new_image_filename)
    new_image = Image.fromarray(shuffled_image)
    new_image.save(new_image_path)
    
    new_image_info = image_info.copy()
    new_image_info['file_name'] = new_image_filename
    
    return new_image_info, new_annotations

def main(num_shuffles: int = 5):
    """
    Main function to create shuffled chessboard images.
    
    Args:
        num_shuffles: Number of shuffled versions to create per input image (default: 5)
    """
    coco_json_path = r"c:\Users\tomde\OneDrive\Documentatie - professioneel - opleiding\AI pro 2025-26\Project\project_chess\data\synthetic_data\instances_default_corrected.json"
    images_dir = r"c:\Users\tomde\OneDrive\Documentatie - professioneel - opleiding\AI pro 2025-26\Project\project_chess\data\synthetic_data"
    output_dir = r"c:\Users\tomde\OneDrive\Documentatie - professioneel - opleiding\AI pro 2025-26\Project\project_chess\data\synthetic_data\shuffled_output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading COCO JSON from: {coco_json_path}")
    coco_data = load_coco_json(coco_json_path)
    
    print(f"Creating {num_shuffles} shuffled versions per image...")
    
    # Copy original images
    print("\nCopying original images...")
    for image_info in coco_data['images']:
        original_path = os.path.join(images_dir, image_info['file_name'])
        if os.path.exists(original_path):
            output_path = os.path.join(output_dir, image_info['file_name'])
            if not os.path.exists(output_path):
                shutil.copy2(original_path, output_path)
    
    # Process images - create multiple shuffled versions
    new_images = []
    new_annotations = []
    next_image_id = max([img['id'] for img in coco_data['images']], default=0) + 1
    next_ann_id = max([ann['id'] for ann in coco_data['annotations']], default=0) + 1
    
    for image_info in coco_data['images']:
        print(f"\nProcessing image: {image_info['file_name']}")
        for shuffle_idx in range(num_shuffles):
            print(f"  Creating shuffle {shuffle_idx + 1}/{num_shuffles}...")
            result = process_image(coco_data, image_info, images_dir, output_dir, shuffle_idx)
            
            if result[0] is None:
                continue
            
            new_image_info, new_image_annotations = result
            new_image_info['id'] = next_image_id
            next_image_id += 1
            
            for ann in new_image_annotations:
                ann['id'] = next_ann_id
                ann['image_id'] = new_image_info['id']
                next_ann_id += 1
            
            new_images.append(new_image_info)
            new_annotations.extend(new_image_annotations)
    
    # Create output COCO data
    new_coco_data = {
        'licenses': coco_data['licenses'],
        'info': coco_data['info'],
        'categories': coco_data['categories'],
        'images': coco_data['images'] + new_images,
        'annotations': coco_data['annotations'] + new_annotations
    }
    
    output_json_path = os.path.join(output_dir, 'instances_shuffled.json')
    print(f"\nSaving shuffled COCO JSON to: {output_json_path}")
    save_coco_json(new_coco_data, output_json_path)
    
    print(f"\nDone! Created {len(new_images)} shuffled images ({num_shuffles} per input image).")
    print(f"Total images: {len(new_coco_data['images'])}")
    print(f"Total annotations: {len(new_coco_data['annotations'])}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create shuffled chessboard images while maintaining checkerboard pattern.')
    parser.add_argument('--num-shuffles', type=int, default=5,
                       help='Number of shuffled versions to create per input image (default: 5)')
    args = parser.parse_args()
    
    main(num_shuffles=args.num_shuffles)

