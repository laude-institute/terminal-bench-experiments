#!/usr/bin/env python3
"""
Test different TraceParser formats with both LLM and embedding classifiers.
"""

import sys
from pathlib import Path
import json

# Add src and scripts directories to path

BASE_DIR = Path(__file__).parent.parent


sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "scripts"))

from trace_parser import TraceParser

traces_dir = BASE_DIR / "traces"


def test_reading_trace(trial_id: str):
    """Test different TraceParser formats with both classification approaches."""
    
    print(f"Testing different formats for trial: {trial_id}")
    print("=" * 80)


    # Parse the trace
    parser = TraceParser(traces_dir)
    trace = parser.parse_trace(trial_id)
    
    print(f"Task: {trace.get_task_name()}")
    print(f"Agent: {trace.get_agent_name()}")
    print(f"Episodes: {trace.get_episode_count()}")
    print(f"Has Exception: {trace.has_exception()}")
    print()
    
    # Test different content formats
    formats = {
        "Full Context": trace.to_text(include_metadata=True),
        "Episodes Only": trace.to_text(include_metadata=False),
        "Failure Context": trace.get_failure_context(),
        "Episodes Subset (3)": trace.get_episodes_text(max_episodes=3),
        "Commands Only": "\n".join(trace.get_commands_executed())
    }

    for format_name, content in formats.items():
        print(f"\n{format_name.upper()} FORMAT")
        print("-" * 50)
        print(f"Content length: {len(content):,} characters")
        print(content)
        print("End of content")
        print()
    


if __name__ == "__main__":
    # Test with the 7z cracking trial
    trial_id = "0bbfbbc6-7e9e-4f8b-a517-517a04589157"
    
    try:
        test_reading_trace(trial_id)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the traces directory and the trial exists.")