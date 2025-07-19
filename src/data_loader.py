# src/data_loader.py (Corrected version)

import json
import os
import re

def load_video_data(json_path: str, video_filename: str):
    """
    Loads configuration information for a specific video from the JSON task file,
    including the target category.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Error: JSON task file not found at path: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    match = re.search(r'(\d+)', video_filename)
    if not match:
        raise ValueError(f"Could not extract a numeric ID from the video filename '{video_filename}'.")
    video_key = match.group(1)

    if video_key not in data:
        raise ValueError(f"Error: Key '{video_key}' matching video '{video_filename}' not found in JSON file '{os.path.basename(json_path)}'.")

    task_info = data[video_key]

    # Extract all necessary fields
    start_frame = task_info.get('temp_gt', {}).get('begin_fid')
    end_frame = task_info.get('temp_gt', {}).get('end_fid')
    query = task_info.get('sentence', {}).get('description')
    target_category = task_info.get('target_category') # <-- This was the missing piece

    if any(info is None for info in [start_frame, end_frame, query, target_category]):
        raise ValueError(f"Error: Task information for video '{video_key}' is incomplete. Please ensure the JSON contains 'begin_fid', 'end_fid', 'description', and 'target_category'.")

    # The video path comes from the command-line arguments, not the JSON
    video_path = os.path.join("sample_videos", video_filename)

    # Return all 5 values
    return video_path, start_frame, end_frame, query, target_category