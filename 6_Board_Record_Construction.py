import os
import numpy as np
import csv
import re
import shutil
from tqdm import tqdm

def custom_sort_key(file_path):
    """Custom sorting function for special filename formats"""
    filename = os.path.basename(file_path)
    match = re.match(r'frame_(\d+)-(\d+)-(\d+)(?:-(\d+))?', filename)
    if match:
        groups = match.groups()
        parts = [int(g) if g else 0 for g in groups]
        return parts
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

def load_npy_sequence(folder):
    """Load all npy files in a folder, sorted naturally, return (list of ndarray)"""
    npy_files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    npy_paths = [os.path.join(folder, f) for f in npy_files]
    
    # Use custom sorting
    npy_paths = sorted(npy_paths, key=custom_sort_key)
    npy_files = [os.path.basename(p) for p in npy_paths]
    
    print(f"Loading {len(npy_paths)} npy files...")
    boards = []
    for path in tqdm(npy_paths, desc="Loading files"):
        boards.append(np.load(path))
    
    return boards, npy_files

def is_dead(board, pos, color):
    """DFS to determine if a group of this color is dead, return True and group coordinates if dead"""
    visited, group, has_liberty = set(), set(), [False]
    rows, cols = board.shape
    def dfs(r, c):
        visited.add((r,c))
        group.add((r,c))
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            rr, cc = r+dr, c+dc
            if 0<=rr<rows and 0<=cc<cols and (rr,cc) not in visited:
                if board[rr, cc] == 0:
                    has_liberty[0] = True
                elif board[rr, cc] == color:
                    dfs(rr, cc)
    dfs(*pos)
    return not has_liberty[0], group

def get_board_record(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create virtual board and confidence map folders
    virtual_board_dir = os.path.join(output_folder, "virtual_boards")
    confidence_dir = os.path.join(output_folder, "confidence_maps")
    if not os.path.exists(virtual_board_dir):
        os.makedirs(virtual_board_dir)
    if not os.path.exists(confidence_dir):
        os.makedirs(confidence_dir)
    
    boards, fnames = load_npy_sequence(input_folder)
    N = boards[0].shape[0]
    move_history = []     # (move_no, color, (r,c))
    conf_map = np.zeros((N, N), dtype=int)
    curr = np.zeros((N, N), dtype=int)
    move_no = 0

    print("Analyzing board changes...")
    for i, (board, fname) in enumerate(tqdm(zip(boards, fnames), total=len(boards), desc="Processing boards")):
        # Get current stone coordinates
        black_set = set(zip(*np.where(board == 1)))
        white_set = set(zip(*np.where(board == 2)))
        stone_set = black_set | white_set
        
        # Get base filename (without extension)
        basename = os.path.splitext(fname)[0]
        
        # Initialize
        if i == 0:
            for pos in stone_set:
                color = board[pos]
                curr[pos] = color
                conf_map[pos] = 50
                move_history.append({
                    "step": move_no, 
                    "color": color, 
                    "pos": pos, 
                    "action": "init", 
                    "frame": fname, 
                    "deleted": False,
                    "virtual_board_file": fname,
                    "confidence_file": f"{basename}_conf.npy"
                })
                move_no += 1
                
            # Save virtual board and confidence map
            virtual_board_path = os.path.join(virtual_board_dir, fname)
            confidence_path = os.path.join(confidence_dir, f"{basename}_conf.npy")
            np.save(virtual_board_path, curr.copy())
            np.save(confidence_path, conf_map.copy())
            continue
            
        # Stone change processing
        prev_stones = set(zip(*np.where(curr > 0)))
        removed = prev_stones - stone_set
        added = stone_set - prev_stones
        
        # Prepare virtual board and confidence map paths for this frame
        virtual_board_path = os.path.join(virtual_board_dir, fname)
        confidence_path = os.path.join(confidence_dir, f"{basename}_conf.npy")
        
        # Process removed stones
        for pos in removed:
            c = curr[pos]
            dead, group = is_dead(curr, pos, c)
            if dead:
                for p in group:
                    curr[p] = 0
                    conf_map[p] = 0
                    move_history.append({
                        "step": move_no, 
                        "color": c, 
                        "pos": p, 
                        "action": "capture", 
                        "frame": fname, 
                        "deleted": False,
                        "virtual_board_file": fname,
                        "confidence_file": f"{basename}_conf.npy"
                    })
                    move_no += 1
            else:
                # Ensure confidence value doesn't go below 0
                conf_map[pos] = max(0, conf_map[pos] - 1)
                if conf_map[pos] == 0:
                    curr[pos] = 0
                    # Soft delete, find closest add record
                    for rec in reversed(move_history):  # Search backwards
                        if (rec["action"] == "add" and rec["pos"] == pos and rec["color"] == c and not rec.get("deleted", False)):
                            rec["deleted"] = True
                            break
                    
        # Process additions
        if len(added) >= 1:
            conf = 50 if len(added) == 1 else 150
            for pos in added:
                color = board[pos]
                curr[pos] = color
                conf_map[pos] = conf
                move_history.append({
                    "step": move_no, 
                    "color": color, 
                    "pos": pos, 
                    "action": "add", 
                    "frame": fname, 
                    "deleted": False,
                    "virtual_board_file": fname,
                    "confidence_file": f"{basename}_conf.npy"
                })
                move_no += 1
        
        # Save virtual board and confidence map for this frame
        np.save(virtual_board_path, curr.copy())
        np.save(confidence_path, conf_map.copy())
    
    return move_history

# Example usage
if __name__ == "__main__":
    import time
    start_time = time.time()
    
    input_folder = "./go_board_cut_pred_h"
    output_folder = "./go_board_cut_pred_h_results"
    moves = get_board_record(input_folder, output_folder)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time:.2f} seconds")