import os
import re
import numpy as np
import csv
from collections import defaultdict

def custom_sort_key(file_path):
    filename = os.path.basename(file_path)
    match = re.match(r'frame_(\d+)-(\d+)-(\d+)(?:-(\d+))?', filename)
    if match:
        groups = match.groups()
        parts = [int(g) if g else 0 for g in groups]
        return parts
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

def expand_region(array, target_value=3):
    original = array.copy()
    modified = array.copy()
    rows, cols = array.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(rows):
        for c in range(cols):
            if original[r, c] == target_value:
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        modified[nr, nc] = target_value
    return modified

def detect_board_changes(prev_board, curr_board):
    """Detect changes between two board states, return list of change descriptions"""
    changes = []
    
    # Map values to descriptions - corrected: 1=Black, 2=White
    value_to_desc = {0: "Empty", 1: "Black", 2: "White", 3: "Occlusion"}
    
    # Check each position
    for i in range(prev_board.shape[0]):
        for j in range(prev_board.shape[1]):
            if prev_board[i, j] != curr_board[i, j]:
                from_val = value_to_desc[prev_board[i, j]]
                to_val = value_to_desc[curr_board[i, j]]
                changes.append(f"({i},{j},{from_val}->{to_val})")
    
    return changes

def count_stone_changes(prev_board, curr_board):
    """Count new black and white stones"""
    new_black_count = 0
    new_white_count = 0
    
    for i in range(prev_board.shape[0]):
        for j in range(prev_board.shape[1]):
            # Empty to black
            if prev_board[i, j] == 0 and curr_board[i, j] == 1:
                new_black_count += 1
            # Empty to white
            elif prev_board[i, j] == 0 and curr_board[i, j] == 2:
                new_white_count += 1
    
    return new_black_count, new_white_count

def has_occlusion(board):
    """Check if the board contains occlusion (grid with value 3)"""
    return 3 in board

def sequential_board_update_and_save(src_dir, dst_dir, changes_file, expand_times=1):
    """
    Process board and save results
    
    Parameters:
    src_dir: Source file directory
    dst_dir: Target file directory
    changes_file: Changes record file path
    expand_times: Number of times to expand occlusion area, 0 means no expansion, 1 means expand once, 2 means expand twice
    """
    os.makedirs(dst_dir, exist_ok=True)
    npy_files = [f for f in os.listdir(src_dir) if f.endswith('.npy')]
    npy_files_sorted = sorted([os.path.join(src_dir, f) for f in npy_files], key=custom_sort_key)
    print(f"Found {len(npy_files_sorted)} sorted npy files to process.")
    print(f"Expansion times set to: {expand_times}")

    # Statistics variables
    multi_stone_frames = 0  # Number of frames with 2+ new stones
    frames_with_changes = 0  # Total frames with changes
    total_changes = 0  # Total number of changes
    orig_frames_with_occlusion = 0  # Number of original files with occlusion
    expanded_frames_with_occlusion = 0  # Number of frames with occlusion after expansion
    change_types = defaultdict(int)  # Statistics of different change types

    # Ensure first frame is not all zeros
    first_valid_board = None
    for path in npy_files_sorted:
        arr = np.load(path)
        if np.sum(arr) > 0:  # If not all zeros
            first_valid_board = arr.copy()
            break
    
    if first_valid_board is None:
        print("Error: No valid non-zero board found!")
        return
    
    last_saved_board = None
    save_count = 0
    
    # Create CSV file and write header
    with open(changes_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Saved File", "Original File", "Changes", "Black Added", "White Added", "Multiple Stones", "Original Has Occlusion"])
        
        for i, src_path in enumerate(npy_files_sorted):
            orig_fname = os.path.basename(src_path)
            arr = np.load(src_path)
            
            # Check if original file has occlusion
            original_has_occlusion = has_occlusion(arr)
            if original_has_occlusion:
                orig_frames_with_occlusion += 1
            
            # Expand occlusion area based on setting
            if expand_times > 0:
                for _ in range(expand_times):
                    arr = expand_region(arr, target_value=3)
            
            # Check if occlusion remains after expansion
            if has_occlusion(arr):
                expanded_frames_with_occlusion += 1
                
            # Handle first frame issue
            if i == 0 and np.sum(arr) == 0:
                print("First frame is empty, using next valid frame")
                arr = first_valid_board.copy()
            
            # Replace occlusion (3) with previous frame for non-first frames
            if last_saved_board is not None:
                arr = arr.copy()
                arr[arr == 3] = last_saved_board[arr == 3]
            else:
                # First frame
                last_saved_board = arr.copy()
            
            # Verify no occlusion remains after processing
            if has_occlusion(arr):
                print(f"Warning: Board {orig_fname} still contains occlusion after processing, this should not happen!")
            
            # Save current board state using original filename
            dst_path = os.path.join(dst_dir, orig_fname)
            
            # Detect board changes
            changes = []
            has_changes = False
            new_black = 0
            new_white = 0
            multiple_stones = "No"
            
            if last_saved_board is not None:
                changes = detect_board_changes(last_saved_board, arr)
                has_changes = len(changes) > 0
                
                # Count new black and white stones
                new_black, new_white = count_stone_changes(last_saved_board, arr)
                
                # Check if 2+ stones added in one frame
                if new_black + new_white >= 2:
                    multiple_stones = "Yes"
                    multi_stone_frames += 1
            
            # Save regardless of changes
            np.save(dst_path, arr)
            
            # Update statistics
            if has_changes:
                frames_with_changes += 1
                total_changes += len(changes)
                
                # Count change types
                for change in changes:
                    # Extract change type, e.g., Empty->Black
                    change_match = re.search(r'(\w+)->(\w+)', change)
                    if change_match:
                        from_val, to_val = change_match.groups()
                        change_type = f"{from_val}->{to_val}"
                        change_types[change_type] += 1
            
            # Write change record
            if has_changes:
                changes_str = "; ".join(changes)
                writer.writerow([save_count, orig_fname, orig_fname, f'"{changes_str}"', new_black, new_white, multiple_stones, "Yes" if original_has_occlusion else "No"])
                print(f"Saved board with changes: {dst_path} - {len(changes)} changes - New Black: {new_black}, White: {new_white}, Multiple Stones: {multiple_stones}, Original Frame Has Occlusion: {'Yes' if original_has_occlusion else 'No'}")
            else:
                writer.writerow([save_count, orig_fname, orig_fname, '""', 0, 0, "No", "Yes" if original_has_occlusion else "No"])
                print(f"Saved board without changes: {dst_path}, Original Frame Has Occlusion: {'Yes' if original_has_occlusion else 'No'}")
            
            # Update last saved board and counter
            last_saved_board = arr.copy()
            save_count += 1
    
    # Calculate occlusion frame percentages
    orig_occlusion_percentage = (orig_frames_with_occlusion / save_count * 100) if save_count > 0 else 0
    expanded_occlusion_percentage = (expanded_frames_with_occlusion / save_count * 100) if save_count > 0 else 0
    
    # Output statistics
    print("\n====== Processing Statistics ======")
    print(f"Total Frames: {save_count}")
    print(f"Frames with Changes: {frames_with_changes}")
    print(f"Frames with 2+ Stones Added: {multi_stone_frames}")
    print(f"Total Changes: {total_changes}")
    print(f"Original Files with Occlusion: {orig_frames_with_occlusion} ({orig_occlusion_percentage:.2f}%)")
    print(f"Frames with Occlusion After Expansion: {expanded_frames_with_occlusion} ({expanded_occlusion_percentage:.2f}%)")
    print("(Occlusion areas in all frames have been replaced with previous frame data, so no occlusion should remain)")
    print("\nChange Type Statistics:")
    for change_type, count in sorted(change_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {change_type}: {count} times")

# Example usage
src_dir = "./go_board_cut_pred"
dst_dir = "./go_board_cut_pred_h"
changes_file = "./go_board_cut_pred_h/changes.csv"

# Usage: Set expand_times parameter to 0, 1, or 2
# 0: No occlusion area expansion
# 1: Expand occlusion area once
# 2: Expand occlusion area twice
sequential_board_update_and_save(src_dir, dst_dir, changes_file, expand_times=1)