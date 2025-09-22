#!/usr/bin/env python3
"""
TraceParser class for reading and parsing Terminal Bench traces.

This module provides the TraceParser class for loading trace data from
downloaded trace directories and parsing their contents into structured format.

## Terminal Bench Trace Structure

Terminal Bench traces are stored in directories with the following structure:

```
traces/
├── {trial_id}/                    # UUID trial identifier
│   ├── result.json                # Trial execution results and metadata
│   ├── config.json                # Trial configuration parameters
│   ├── exception.txt              # Exception information (if trial failed)
│   ├── agent/                     # Agent execution episodes
│   │   ├── episode-0/
│   │   │   ├── prompt.txt         # Input prompt to agent
│   │   │   └── response.txt       # Agent's response (JSON format)
│   │   ├── episode-1/
│   │   │   ├── prompt.txt
│   │   │   └── response.txt
│   │   └── ...
│   └── verifier/                  # Task verification data (optional)
│       ├── output.json
│       └── logs.txt
```

## Data Content Description

### result.json
Contains trial execution metadata:
- `trial_id`: Unique identifier for this trial
- `task_name`: Name of the Terminal Bench task (e.g., "gpt2-codegolf", "crack-7z-hash")
- `agent_name`: Agent implementation name (e.g., "terminus-2")
- `reward`: Final reward score (float, typically 0.0-1.0, or null if failed)
- `created_at`: Timestamp of trial execution
- `task_checksum`: Hash of the task definition
- `config`: Nested configuration object

### config.json
Contains trial configuration parameters:
- Task parameters (timeout, environment settings)
- Agent parameters (model configuration, reasoning settings)
- Orchestrator settings (retry policies, resource limits)

### exception.txt
Present only if the trial encountered an exception during execution.
Contains the full exception traceback and error message.

### agent/episode-N/
Each episode represents one interaction cycle between the orchestrator and agent:

#### prompt.txt
The input prompt sent to the agent, typically containing:
- Task description and objectives
- Current terminal state (screen content, working directory)
- Previous command outputs
- JSON format specification for agent responses
- Execution constraints (timeouts, allowed operations)

#### response.txt
The agent's response in JSON format, typically containing:
- `analysis`: Agent's understanding of current situation
- `plan`: Agent's intended next steps
- `commands`: Array of terminal commands to execute
- `task_complete`: Boolean indicating if agent believes task is done

## Model Input Recommendations

When feeding trace data to classification models, consider these approaches:

### 1. Full Trace Context (Recommended for LLMs)
Use `TraceData.to_text(include_metadata=True)` which provides:
- Complete trial metadata (task, agent, model, reward)
- All episode interactions (prompts and responses)
- Exception information if present
- Structured format that preserves temporal order

This format is ideal for:
- Large language models (GPT, Claude) that can handle long contexts
- Detailed failure analysis requiring full execution context
- Cases where you need to understand the complete agent reasoning process

### 2. Episodes-Only Context (For focused analysis)
Use `TraceData.to_text(include_metadata=False)` which provides:
- Only the agent episodes (prompts and responses)
- Cleaner format without metadata noise
- Focus on agent behavior and reasoning

This format is ideal for:
- Models with limited context windows
- Analysis focused on agent reasoning patterns
- Embedding-based similarity matching

### 3. Individual Episode Analysis
Access individual episodes via `TraceData.episodes` for:
- Fine-grained analysis of specific interaction points
- Sequential analysis of agent behavior over time
- Training data for episode-level classification

### 4. Metadata-Only Analysis
Use the structured metadata fields for:
- Statistical analysis across many trials
- Filtering and grouping by task/agent/model
- Quick triage before deeper analysis

## Classification Input Guidelines

### For Failure Classification:
1. **Include task context**: Task name helps the model understand expected behavior
2. **Include exceptions**: Critical for understanding failure modes
3. **Preserve episode order**: Temporal sequence reveals behavior patterns
4. **Include terminal outputs**: Command results show what actually happened vs. intended

### For Success Pattern Analysis:
1. **Focus on final episodes**: Later episodes show successful completion patterns
2. **Include reward information**: Helps distinguish partial vs. complete success
3. **Compare with task objectives**: Use task description to validate success

### For Agent Behavior Analysis:
1. **Examine response patterns**: Look at analysis/plan consistency
2. **Track command evolution**: How commands change across episodes
3. **Monitor context awareness**: How well agent maintains state

## Performance Considerations

- **Large traces**: Some trials have 10+ episodes with long terminal outputs
- **Memory usage**: Full traces can be 50KB+ of text
- **Token limits**: Consider truncation strategies for very long traces
- **Parsing cost**: Cache parsed TraceData objects for repeated analysis

## Example Usage for Model Input

```python
# For comprehensive LLM analysis
parser = TraceParser("traces")
trace = parser.parse_trace(trial_id)
full_context = trace.to_text(include_metadata=True)
# Send full_context to LLM for classification

# For embedding-based similarity
episodes_only = trace.to_text(include_metadata=False)
# Use episodes_only for embedding computation

# For structured analysis
task_name = trace.get_task_name()
has_errors = trace.has_exception()
episode_count = trace.get_episode_count()
# Use metadata for filtering and statistical analysis
```
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


BASE_DIR = Path(__file__).parent.parent

@dataclass
class EpisodeData:
    """Data structure for a single agent episode."""
    episode_number: int
    prompt: str
    response: str
    

@dataclass
class TraceData:
    """Data structure for a complete trace."""
    trial_id: str
    result: Optional[Dict[str, Any]]
    config: Optional[Dict[str, Any]]
    exception: Optional[str]
    episodes: List[EpisodeData]
    verifier_data: Optional[Dict[str, Any]]
    
    def get_task_name(self) -> str:
        """Extract task name from trace data."""
        if self.result and 'task_name' in self.result:
            return self.result['task_name']
        if self.config and 'task' in self.config and 'name' in self.config['task']:
            return self.config['task']['name']
        return "Unknown Task"
    
    def get_agent_name(self) -> str:
        """Extract agent name from trace data."""
        if self.result and 'agent_name' in self.result:
            return self.result['agent_name']
        if self.config and 'agent' in self.config and 'name' in self.config['agent']:
            return self.config['agent']['name']
        return "Unknown Agent"
    
    def get_model_name(self) -> str:
        """Extract model name from trace data."""
        if self.result and 'model_name' in self.result:
            return self.result['model_name']
        if self.config and 'model' in self.config and 'name' in self.config['model']:
            return self.config['model']['name']
        return "Unknown Model"
    
    def get_reward(self) -> Optional[float]:
        """Extract reward from trace data."""
        if self.result and 'verifier_result' in self.result:
            return self.result['verifier_result']['reward']
        return None
    
    def has_exception(self) -> bool:
        """Check if trace has exception information."""
        return self.exception is not None and len(self.exception.strip()) > 0
    
    def get_episode_count(self) -> int:
        """Get number of episodes in the trace."""
        return len(self.episodes)
    
    def to_text(self, include_metadata: bool = False) -> str:
        """Convert trace data to text format for analysis."""
        content = []
        
        if include_metadata:
            content.append(f"Trial ID: {self.trial_id}")
            content.append(f"Task: {self.get_task_name()}")
            content.append(f"Agent: {self.get_agent_name()}")
            content.append(f"Model: {self.get_model_name()}")
            if self.get_reward() is not None:
                content.append(f"Reward: {self.get_reward()}")
            content.append("")
        
        # Add configuration if available
        if self.config and include_metadata:
            content.append("=== CONFIGURATION ===")
            content.append(json.dumps(self.config, indent=2))
            content.append("")
        
        # Add result data if available
        if self.result and include_metadata:
            content.append("=== RESULT ===")
            content.append(json.dumps(self.result, indent=2))
            content.append("")
        
        # Add exception if present
        if self.has_exception():
            content.append("=== EXCEPTION ===")
            content.append(self.exception)
            content.append("")
        
        # Add episodes
        if self.episodes:
            content.append("=== AGENT EPISODES ===")
            for episode in self.episodes:
                content.append(f"\n--- Episode {episode.episode_number} ---")
                content.append("PROMPT:")
                content.append(episode.prompt)
                content.append("\nRESPONSE:")
                content.append(episode.response)
                content.append("")
        
        # Add verifier data if available
        if self.verifier_data and include_metadata:
            content.append("=== VERIFIER ===")
            content.append(json.dumps(self.verifier_data, indent=2))
            content.append("")
        
        return "\n".join(content)
    
    def get_episodes_text(self, max_episodes: Optional[int] = None) -> str:
        """
        Get just the episodes content as text, useful for focused analysis.
        
        Args:
            max_episodes: Limit to first N episodes (useful for very long traces)
            
        Returns:
            Episodes formatted as text
        """
        if not self.episodes:
            return ""
        
        content = []
        episodes_to_process = self.episodes[:max_episodes] if max_episodes else self.episodes
        
        for episode in episodes_to_process:
            content.append(f"--- Episode {episode.episode_number} ---")
            content.append("PROMPT:")
            content.append(episode.prompt)
            content.append("\nRESPONSE:")
            content.append(episode.response)
            content.append("")
        
        return "\n".join(content)
    
    def get_failure_context(self) -> str:
        """
        Get text optimized for failure classification models.
        Includes task context, exceptions, and episodes but minimal metadata.
        
        Returns:
            Failure-focused text content
        """
        content = []
        
        # Essential context for failure analysis
        content.append(f"Task: {self.get_task_name()}")
        content.append(f"Agent: {self.get_agent_name()}")
        if self.get_reward() is not None:
            content.append(f"Final Reward: {self.get_reward()}")
        content.append("")
        
        # Include exceptions (critical for failure analysis)
        if self.has_exception():
            content.append("=== EXCEPTION ===")
            content.append(self.exception)
            content.append("")
        
        # Include episodes (show the actual execution)
        if self.episodes:
            content.append("=== EXECUTION TRACE ===")
            content.append(self.get_episodes_text())
        
        return "\n".join(content)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics about the trace for quick analysis.
        
        Returns:
            Dictionary with trace statistics
        """
        stats = {
            "trial_id": self.trial_id,
            "task_name": self.get_task_name(),
            "agent_name": self.get_agent_name(),
            "model_name": self.get_model_name(),
            "reward": self.get_reward(),
            "episode_count": self.get_episode_count(),
            "has_exception": self.has_exception(),
            "has_config": self.config is not None,
            "has_result": self.result is not None,
            "has_verifier_data": self.verifier_data is not None,
        }
        
        # Add episode-level stats
        if self.episodes:
            total_prompt_chars = sum(len(ep.prompt) for ep in self.episodes)
            total_response_chars = sum(len(ep.response) for ep in self.episodes)
            stats.update({
                "total_prompt_chars": total_prompt_chars,
                "total_response_chars": total_response_chars,
                "avg_prompt_chars": total_prompt_chars // len(self.episodes),
                "avg_response_chars": total_response_chars // len(self.episodes),
                "first_episode_num": self.episodes[0].episode_number,
                "last_episode_num": self.episodes[-1].episode_number,
            })
        
        return stats
    
    def get_commands_executed(self) -> List[str]:
        """
        Extract all commands that were executed by the agent.
        Useful for command pattern analysis.
        
        Returns:
            List of command strings extracted from agent responses
        """
        commands = []
        
        for episode in self.episodes:
            try:
                # Parse the JSON response to extract commands
                response_data = json.loads(episode.response)
                if "commands" in response_data:
                    episode_commands = response_data["commands"]
                    for cmd in episode_commands:
                        if isinstance(cmd, dict) and "keystrokes" in cmd:
                            # Extract the actual command from keystrokes
                            keystroke = cmd["keystrokes"].strip()
                            if keystroke.endswith("\n"):
                                keystroke = keystroke[:-1]  # Remove trailing newline
                            if keystroke:  # Only add non-empty commands
                                commands.append(keystroke)
                        elif isinstance(cmd, str):
                            commands.append(cmd)
            except (json.JSONDecodeError, KeyError):
                # Skip episodes with malformed JSON responses
                continue
        
        return commands
    
    def get_task_description(self) -> str:
        """
        Extract task description from the first episode prompt.
        Useful for understanding what the agent was supposed to do.
        
        Returns:
            Task description string, or empty if not found
        """
        if not self.episodes:
            return ""
        
        # Task description is typically in the first episode prompt
        first_prompt = self.episodes[0].prompt
        
        # Look for "Task Description:" section
        lines = first_prompt.split('\n')
        description_lines = []
        in_description = False
        
        for line in lines:
            if line.strip().startswith("Task Description:"):
                in_description = True
                continue
            elif in_description and line.strip() and not line.startswith(" "):
                # End of description section
                break
            elif in_description:
                description_lines.append(line)
        
        return '\n'.join(description_lines).strip()


class TraceParser:
    """Parser for Terminal Bench trace files."""
    
    def __init__(self, traces_dir: str = "traces"):
        """Initialize the trace parser.
        
        Args:
            traces_dir: Directory containing trace subdirectories
        """
        self.traces_dir = Path(traces_dir).resolve()
        print(f"Traces directory: {self.traces_dir}")
        if not self.traces_dir.exists():
            raise FileNotFoundError(f"Traces directory not found: {self.traces_dir}")
    
    def list_available_trials(self) -> List[str]:
        """List all available trial IDs in the traces directory.
        
        Returns:
            List of trial ID strings
        """
        trial_ids = []
        for item in self.traces_dir.iterdir():
            if item.is_dir() and self._is_valid_trial_dir(item):
                trial_ids.append(item.name)
        return sorted(trial_ids)
    
    def _is_valid_trial_dir(self, path: Path) -> bool:
        """Check if a directory contains valid trial data.
        
        Args:
            path: Path to check
            
        Returns:
            True if directory contains trial files
        """
        # Check for at least one of the expected files
        expected_files = ['result.json', 'config.json', 'exception.txt']
        has_file = any((path / f).exists() for f in expected_files)
        
        # Check for agent directory
        has_agent_dir = (path / 'agent').exists()
        
        return has_file or has_agent_dir
    
    def parse_trace(self, trial_id: str) -> TraceData:
        """Parse a complete trace for the given trial ID.
        
        Args:
            trial_id: Trial ID to parse
            
        Returns:
            TraceData object containing parsed trace information
            
        Raises:
            FileNotFoundError: If trial directory doesn't exist
        """
        trace_dir = self.traces_dir / trial_id
        
        if not trace_dir.exists():
            raise FileNotFoundError(f"Trial directory not found: {trace_dir}")
        
        # Parse result.json
        result = self._load_json_file(trace_dir / "result.json")
        
        # Parse config.json
        config = self._load_json_file(trace_dir / "config.json")
        
        # Parse exception.txt
        exception = self._load_text_file(trace_dir / "exception.txt")
        
        # Parse agent episodes
        episodes = self._parse_episodes(trace_dir / "agent")
        
        # Parse verifier data
        verifier_data = self._parse_verifier_data(trace_dir / "verifier")
        
        return TraceData(
            trial_id=trial_id,
            result=result,
            config=config,
            exception=exception,
            episodes=episodes,
            verifier_data=verifier_data
        )
    
    def _load_json_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load JSON file if it exists.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON data or None if file doesn't exist
        """
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Warning: Could not parse JSON file {file_path}: {e}")
            return None
    
    def _load_text_file(self, file_path: Path) -> Optional[str]:
        """Load text file if it exists.
        
        Args:
            file_path: Path to text file
            
        Returns:
            File content or None if file doesn't exist
        """
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError as e:
            print(f"Warning: Could not read text file {file_path}: {e}")
            return None
    
    def _parse_episodes(self, agent_dir: Path) -> List[EpisodeData]:
        """Parse agent episodes from the agent directory.
        
        Args:
            agent_dir: Path to agent directory
            
        Returns:
            List of EpisodeData objects
        """
        episodes = []
        
        if not agent_dir.exists():
            return episodes
        
        # Find all episode directories
        episode_dirs = [d for d in agent_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('episode-')]
        
        # Sort by episode number
        episode_dirs.sort(key=lambda x: int(x.name.split('-')[1]))
        
        for episode_dir in episode_dirs:
            episode_num = int(episode_dir.name.split('-')[1])
            
            # Read prompt and response
            prompt = self._load_text_file(episode_dir / "prompt.txt") or ""
            response = self._load_text_file(episode_dir / "response.txt") or ""
            
            episodes.append(EpisodeData(
                episode_number=episode_num,
                prompt=prompt,
                response=response
            ))
        
        return episodes
    
    def _parse_verifier_data(self, verifier_dir: Path) -> Optional[Dict[str, Any]]:
        """Parse verifier data if available.
        
        Args:
            verifier_dir: Path to verifier directory
            
        Returns:
            Verifier data dictionary or None
        """
        if not verifier_dir.exists():
            return None
        
        verifier_data = {}
        
        # Look for common verifier files
        for file_path in verifier_dir.rglob("*.json"):
            rel_path = file_path.relative_to(verifier_dir)
            data = self._load_json_file(file_path)
            if data:
                verifier_data[str(rel_path)] = data
        
        for file_path in verifier_dir.rglob("*.txt"):
            rel_path = file_path.relative_to(verifier_dir)
            data = self._load_text_file(file_path)
            if data:
                verifier_data[str(rel_path)] = data
        
        return verifier_data if verifier_data else None
    
    def get_trace_summary(self, trial_id: str) -> Dict[str, Any]:
        """Get a summary of trace information without full parsing.
        
        Args:
            trial_id: Trial ID to summarize
            
        Returns:
            Dictionary with trace summary information
        """
        trace_dir = self.traces_dir / trial_id
        
        if not trace_dir.exists():
            return {"error": f"Trial directory not found: {trace_dir}"}
        
        summary = {
            "trial_id": trial_id,
            "has_result": (trace_dir / "result.json").exists(),
            "has_config": (trace_dir / "config.json").exists(),
            "has_exception": (trace_dir / "exception.txt").exists(),
            "has_agent_data": (trace_dir / "agent").exists(),
            "has_verifier_data": (trace_dir / "verifier").exists(),
            "episode_count": 0
        }
        
        # Count episodes
        agent_dir = trace_dir / "agent"
        if agent_dir.exists():
            episode_dirs = [d for d in agent_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('episode-')]
            summary["episode_count"] = len(episode_dirs)
        
        # Get basic metadata from result if available
        result = self._load_json_file(trace_dir / "result.json")
        if result:
            summary.update({
                "task_name": result.get("task_name", "Unknown"),
                "agent_name": result.get("agent_name", "Unknown"),
                "reward": result.get("reward")
            })
        
        return summary


if __name__ == "__main__":
    # Example usage
    parser = TraceParser(BASE_DIR / "traces")
    
    trial_id = "0d4d39d9-e67d-4610-8785-bab0ddbb2d81"
    
    try:
        # Get summary
        summary = parser.get_trace_summary(trial_id)
        print(f"Trace Summary: {json.dumps(summary, indent=2)}")
        
        # Parse full trace
        trace = parser.parse_trace(trial_id)
        print(f"\nTask: {trace.get_task_name()}")
        print(f"Agent: {trace.get_agent_name()}")
        print(f"Episodes: {trace.get_episode_count()}")
        print(f"Has Exception: {trace.has_exception()}")
        
        # Convert to text for analysis:
        text_content = trace.to_text(include_metadata=False)
        print(f"\nText content length: {len(text_content)} characters")
        print(f"Text content:\n\n{text_content}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        
        # List available trials
        available = parser.list_available_trials()
        print(f"Available trials: {available[:5]}...")  # Show first 5