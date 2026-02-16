#!/usr/bin/env python3
"""
Simple script to download essential files and get episode counts for terminus-2 trials.
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

def download_file(client, trial_id, file_path, output_path):
    """Download a single file from Supabase storage"""
    try:
        response = client.storage.from_("trials").download(f"{trial_id}/{file_path}")
        with open(output_path, 'wb') as f:
            f.write(response)
        return True
    except:
        return False

def count_episodes(client, trial_id):
    """Count episodes by listing directories in Supabase storage"""
    try:
        response = client.storage.from_("trials").list(f"{trial_id}/agent")
        if response:
            return len([item for item in response if item.get('name', '').startswith('episode-')])
        return 0
    except:
        return -1

def process_trial(client, trial, output_dir):
    """Download essential files and get episode count for one trial"""
    trial_id = trial["id"]
    trial_dir = output_dir / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    # Download essential files
    files_to_download = [
        ("config.json", "config.json"),
        ("result.json", "result.json"), 
        ("agent/recording.cast", "recording.cast")
    ]
    
    downloaded = 0
    for supabase_path, local_filename in files_to_download:
        output_path = trial_dir / local_filename
        if download_file(client, trial_id, supabase_path, output_path):
            downloaded += 1
    
    # Get episode count
    episode_count = count_episodes(client, trial_id)
    
    # Save episode count to file
    if episode_count >= 0:
        with open(trial_dir / "episode_count.txt", 'w') as f:
            f.write(f"{episode_count}\n")
    
    # Extract trial metadata for JSON
    trial_model = trial.get("trial_model", {})
    model_name = trial_model.get("model_name", "Unknown") if trial_model else "Unknown"
    
    verifier_result = trial.get("verifier_result") or {}
    reward = verifier_result.get("reward", 0) if isinstance(verifier_result, dict) else 0
    
    trial_data = {
        "trial_id": trial_id,
        "trial_name": trial.get("trial_name", "Unknown"),
        "agent_name": trial.get("agent_name", "Unknown"),
        "agent_version": trial.get("agent_version", "Unknown"),
        "created_at": trial.get("created_at", "Unknown"),
        "task_name": trial.get("task_name", "Unknown"),
        "model_name": model_name,
        "reward": reward,
        "episode_count": episode_count if episode_count >= 0 else None
    }
    
    return downloaded, episode_count, trial_data

def main():
    client = create_client(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ["SUPABASE_PUBLISHABLE_KEY"],
    )
    
    # TODO: Set output directory path for downloaded files
    output_dir = Path(__file__).parent.parent.parent / "terminus2_9-17_essential_files"
    output_dir.mkdir(exist_ok=True)
    
    all_trials = []
    offset = 0
    batch_size = 1000
    
    while True:
        response = (
            client.table("trial")
            .select("*, task!inner(*, dataset_task!inner(*)), trial_model!inner(*)")
            .or_("exception_info.is.null,exception_info->>exception_type.eq.AgentTimeoutError")
            .eq("task.dataset_task.dataset_name", "terminal-bench")                              
            .eq("task.dataset_task.dataset_version", "2.0")                                      
            .eq("agent_name", "terminus-2")                                                      
            .gte("created_at", "2025-09-17T01:13:33.950824+00:00")                             
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        
        batch_trials = response.data
        if not batch_trials:
            break
            
        all_trials.extend(batch_trials)
        
        if len(batch_trials) < batch_size:
            break
            
        offset += batch_size
    
    if not all_trials:
        return
    
    success_count = 0
    total_files = 0
    all_trial_data = []
    
    for trial in all_trials:
        trial_id = trial["id"]
        downloaded, episode_count, trial_data = process_trial(client, trial, output_dir)
        total_files += downloaded
        all_trial_data.append(trial_data)
        
        if downloaded == 3 and episode_count >= 0:
            success_count += 1
        
        time.sleep(0.1)
    
    # Save episode counts JSON
    json_file = output_dir / "episode_counts.json"
    with open(json_file, 'w') as f:
        json.dump(all_trial_data, f, indent=2)
    
    print(f"Processed {len(all_trials)} trials, {success_count} successful, {total_files} files downloaded")

if __name__ == "__main__":
    main()
