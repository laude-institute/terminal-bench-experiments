from typing import Dict, List, Any, Optional
import os
from pathlib import Path
import json
import re
BASE_DIR = Path(__file__).parent.parent


FAILURE_MODE_RUBRIC = {
    "Instruction Misunderstanding": {
        "definition": "The agent misunderstood the natural language task.",
        "indicators": [
            "Executes irrelevant commands",
            "Solves only part of the task",
            "Attempts wrong subgoal"
        ],
        "examples": [
            "Task: copy file A to B -> Agent compresses file instead",
            "Task: run program -> Agent only lists files"
        ]
    },
    "Command Construction Error": {
        "definition": "The agent chose the correct tool but built the command incorrectly.",
        "indicators": [
            "Syntax errors (e.g., 'bash: syntax error')",
            "Invalid flags/options",
            "Misspelled commands ('cmakee' instead of 'cmake')"
        ],
        "examples": [
            "grep -xzf file.txt",
            "ls --not-a-real-flag"
        ]
    },
    "Environment / System Error": {
        "definition": "The command was valid but failed due to environment setup.",
        "indicators": [
            "Dependency missing ('command not found')",
            "File/path errors ('No such file or directory')",
            "Permission issues ('Permission denied')"
        ],
        "examples": [
            "python3: No module named requests",
            "ls: cannot access 'data/': No such file or directory"
        ]
    },
    "Execution Logic Error": {
        "definition": "Commands valid and environment correct, but reasoning flawed.",
        "indicators": [
            "Steps executed in wrong order",
            "Wrong outputs produced",
            "Redundant or looping actions"
        ],
        "examples": [
            "Runs 'make install' before 'make'",
            "Deletes source file before compilation"
        ]
    },
    "Memory / Context Failure": {
        "definition": "The agent lost track of its prior state or actions.",
        "indicators": [
            "Repeats same command with no progress",
            "Refers to nonexistent files",
            "Switches to wrong working directory"
        ],
        "examples": [
            "Creates results.txt but later runs 'cat output.txt'",
            "Repeats 'ls' endlessly"
        ]
    },
    "Error Recovery Failure": {
        "definition": "The agent failed to handle errors effectively.",
        "indicators": [
            "Ignores clear error messages",
            "Retries same failing command",
            "Stops after first failure without retry",
            "Marks task as completed even though it failed"
        ],
        "examples": [
            "command not found -> agent retries same line 3 times",
            "permission denied -> agent gives up"
        ]
    },
    "Timeout / Non-Completion": {
        "definition": "The agent did not complete the task.",
        "indicators": [
            "Trial exceeded max steps/tokens",
            "Agent hangs indefinitely",
            "No output until timeout"
        ],
        "examples": [
            "Agent outputs 'Thinking...' repeatedly until timeout",
            "Process killed due to step budget"
        ]
    }
}


# Output format specification for FailureProcessor classes
OUTPUT_MODE_RUBRIC = {
  "failure_modes": {
    "Instruction Misunderstanding": {
      "score": 0.0,
      "evidence": "",
      "required_skill": "Understand natural language task"
    },
    "Command Construction Error": {
      "score": 0.0,
      "evidence": "Error message: 'bash: syntax error'",
      "required_skill": "Build valid commands with correct syntax"
    },
    "Environment / System Error": {
      "score": 0.0,
      "evidence": "",
      "required_skill": "Check environment dependencies"
    },
    "Execution Logic Error": {
      "score": 0.0,
      "evidence": "",
      "required_skill": "Reason about correct execution order"
    },
    "Memory / Context Failure": {
      "score": 0.0,
      "evidence": "",
      "required_skill": "Maintain context across steps"
    },
    "Error Recovery Failure": {
      "score": 0.0,
      "evidence": "Agent ignored 'command not found' and stopped",
      "required_skill": "Handle errors and retry adaptively"
    },
    "Timeout / Non-Completion": {
      "score": 1.0,
      "evidence": "",
      "required_skill": "Complete within time budget"
    }
  }
}


def make_failure_prompt(trace: str, task_description: Optional[str] = None, **kwargs):
  # Convert FAILURE_MODE_RUBRIC to a string format
    rubric_text = ""
    for mode, details in FAILURE_MODE_RUBRIC.items():
        rubric_text += f"\n{mode}:\n"
        rubric_text += f"  Definition: {details['definition']}\n"
        rubric_text += f"  Indicators: {', '.join(details['indicators'])}\n"
        rubric_text += f"  Examples: {', '.join(details['examples'][:2])}\n"
    
    prompt = f"""You are an expert at analyzing failed terminal/bash execution traces from AI agents attempting to complete tasks.

        FAILURE MODE RUBRIC:
        {rubric_text}

        EXECUTION TRACE:
        {trace}

        TASK DESCRIPTION:
        {task_description} if provided, otherwise "Not provided"

        Analyze this execution trace and classify the final failure that lead to an error into every category from the rubric above.

        For each rubric category, output a JSON object that follows OUTPUT_MODE_RUBRIC exactly:
        - "score": Float between 0.0 and 1.0 (use 0.0 if the category does not apply).
        - "evidence": String. Direct quote or description from the trace that supports this classification (empty string if score = 0.0). It should be the final failure that lead to an error.
        - "required_skill": String. The skill the agent would need to avoid this failure (fill in even if score = 0.0, using the rubric definition).

        Return your answer strictly in this JSON structure:
        {OUTPUT_MODE_RUBRIC}
        
        Only return valid JSON, no explanations outside of it.
        """
    return prompt


TIMEOUT_FAILURE_RUBRIC = {
    "Timeout: Long-running computation": {
        "definition": "Process exceeded time budget",
        "indicators": ["compiling kernel", "running test suite"],
        "examples": ["compiling kernel", "running test suite"],
        "required_skill": "Estimate runtime / chunk long tasks"
    },
    "Timeout: Stuck waiting for I/O": {
        "definition": "Agent blocked on external input or network",
        "indicators": ["waiting for login", "wget hangs"],
        "examples": ["waiting for login", "wget hangs"],
        "required_skill": "Detect stalled I/O, abort & retry"
    },
    "Timeout: Infinite retry loop": {
        "definition": "Agent retries same command until timeout",
        "indicators": ["retrying same command repeatedly"],
        "examples": ["retrying same command repeatedly"],
        "required_skill": "Loop detection / adaptive retry"
    }
}

def make_timeout_prompt(trace: str, task_description: Optional[str] = None):
    """Create a timeout-specific failure analysis prompt."""
    rubric_text = ""
    for mode, details in TIMEOUT_FAILURE_RUBRIC.items():
        rubric_text += f"\n{mode}:\n"
        rubric_text += f"  Definition: {details['definition']}\n"
        rubric_text += f"  Indicators: {', '.join(details['indicators'])}\n"
        rubric_text += f"  Examples: {', '.join(details['examples'])}\n"
        rubric_text += f"  Required Skill: {details['required_skill']}\n"
    
    prompt = f"""You are an expert at analyzing timeout failures in terminal/bash execution traces from AI agents.

TIMEOUT FAILURE MODE RUBRIC:
{rubric_text}

EXECUTION TRACE:
{trace}

TASK DESCRIPTION:
{task_description if task_description else "Not provided"}

Analyze this timeout failure trace and classify it into the timeout-specific categories above.

For each timeout failure category, output a JSON object with:
- "score": Float between 0.0 and 1.0 (use 0.0 if the category does not apply).
- "evidence": String. Direct quote or description from the trace that supports this classification (empty string if score = 0.0).
- "required_skill": String. The skill the agent would need to avoid this failure.

Return your answer strictly in this JSON structure:
{{
  "failure_modes": {{
    {', '.join([f'"{mode}": {{"score": 0.0, "evidence": "", "required_skill": "{details["required_skill"]}"}}' for mode, details in TIMEOUT_FAILURE_RUBRIC.items()])}
  }}
}}

Only return valid JSON, no explanations outside of it.
"""
    return prompt


def make_mast_prompt(trace: str, task_description: str, definitions: Optional[str] = None, examples: Optional[str] = None):

    if definitions is None:
        definitions = open(BASE_DIR / "taxonomies/mast/definitions.txt", "r").read()
    if examples is None:
        examples = open(BASE_DIR / "taxonomies/mast/examples.txt", "r").read()

    prompt = (
        "Below I will provide a multiagent system trace. provide me an analysis of the failure modes and inefficiencies as I will say below. \n"
        "In the traces, analyze the system behaviour."
        "There are several failure modes in multiagent systems I identified. I will provide them below. Tell me if you encounter any of them, as a binary yes or no. \n"
        "Also, give me a one sentence (be brief) summary of the problems with the inefficiencies or failure modes in the trace. Only mark a failure mode if you can provide an example of it in the trace, and specify that in your summary at the end"
        "Also tell me whether the task is successfully completed or not, as a binary yes or no."
        "At the very end, I provide you with the definitions of the failure modes and inefficiencies. After the definitions, I will provide you with examples of the failure modes and inefficiencies for you to understand them better."
        "Tell me if you encounter any of them between the @@ symbols as I will say below, as a binary yes or no."
        "Here are the things you should answer. Start after the @@ sign and end before the next @@ sign (do not include the @@ symbols in your answer):"
        "*** begin of things you should answer *** @@"
        "A. Freeform text summary of the problems with the inefficiencies or failure modes in the trace: <summary>"
        "B. Whether the task is successfully completed or not: <yes or no>"
        "C. Whether you encounter any of the failure modes or inefficiencies:"
        "1.1 Disobey Task Specification: <yes or no>"
        "1.2 Disobey Role Specification: <yes or no>"
        "1.3 Step Repetition: <yes or no>"
        "1.4 Loss of Conversation History: <yes or no>"
        "1.5 Unaware of Termination Conditions: <yes or no>"
        "2.1 Conversation Reset: <yes or no>"
        "2.2 Fail to Ask for Clarification: <yes or no>"
        "2.3 Task Derailment: <yes or no>"
        "2.4 Information Withholding: <yes or no>"
        "2.5 Ignored Other Agent's Input: <yes or no>"
        "2.6 Action-Reasoning Mismatch: <yes or no>"
        "3.1 Premature Termination: <yes or no>"
        "3.2 No or Incorrect Verification: <yes or no>"
        "3.3 Weak Verification: <yes or no>"
        "@@*** end of your answer ***"
        "An example answer is: \n"
        "A. The task is not completed due to disobeying role specification as agents went rogue and started to chat with each other instead of completing the task. Agents derailed and verifier is not strong enough to detect it.\n"
        "B. no \n"
        "C. \n"
        "1.1 no \n"
        "1.2 no \n"
        "1.3 no \n"
        "1.4 no \n"
        "1.5 no \n"
        "1.6 yes \n"
        "2.1 no \n"
        "2.2 no \n"
        "2.3 yes \n"
        "2.4 no \n"
        "2.5 no \n"
        "2.6 yes \n"
        "2.7 no \n"
        "3.1 no \n"
        "3.2 yes \n"
        "3.3 no \n"   
        "Here is the trace: \n"
        f"{trace}"
        "Also, here are the explanations (definitions) of the failure modes and inefficiencies: \n"
        f"{definitions} \n"
        "Here are some examples of the failure modes and inefficiencies: \n"
        f"{examples}"
    )
    return prompt


MAKE_FAILURE_PROMPTS = {
    "base": make_failure_prompt,
    "mast": make_mast_prompt,
    "timeout": make_timeout_prompt
}


def parse_response_base(response: str) -> Dict[str, any]:
    # Strip markdown code blocks if present
    cleaned_response = response.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]
    if cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[3:]
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]
    cleaned_response = cleaned_response.strip()
    
    try:
        result = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON: {e}")
        print(f"Cleaned response was: {repr(cleaned_response)}")
        raise
    
    failure_modes = result.get("failure_modes", [])
    assert all(key in OUTPUT_MODE_RUBRIC['failure_modes'].keys() for key in failure_modes.keys()), "Failure modes must match the OUTPUT_MODE_RUBRIC"
    return failure_modes

def parse_response_mast(response: str) -> Dict[str, any]:
    try:
        failure_modes = {
            '1.1': [], '1.2': [], '1.3': [], '1.4': [], '1.5': [],
            '2.1': [], '2.2': [], '2.3': [], '2.4': [], '2.5': [], '2.6': [],
            '3.1': [], '3.2': [], '3.3': []
        }
        
        # Clean up the response - remove @@ markers if present
        cleaned_response = response.strip()
        if cleaned_response.startswith('@@'):
            cleaned_response = cleaned_response[2:]
        if cleaned_response.endswith('@@'):
            cleaned_response = cleaned_response[:-2]
        
        # Process each failure mode
        for mode in failure_modes.keys():
            # Various patterns to match different response formats
            patterns = [
                # Format with C. prefix and colon
                rf"C\..*?{mode}.*?(yes|no)",
                # Format with just C prefix without dot
                rf"C{mode}\s+(yes|no)",
                # Format with mode directly (with or without spaces)
                rf"{mode}\s*[:]\s*(yes|no)",
                rf"{mode}\s+(yes|no)",
                # Format with newlines
                rf"{mode}\s*\n\s*(yes|no)",
                # Format with C prefix and newlines
                rf"C\.{mode}\s*\n\s*(yes|no)"
            ]
            
            found = False
            for pattern in patterns:
                matches = re.findall(pattern, cleaned_response, re.IGNORECASE | re.DOTALL)
                if matches:
                    # Use the first match
                    value = 1 if matches[0].lower() == 'yes' else 0
                    failure_modes[mode].append(value)
                    found = True
                    break
            
            if not found:
                # If we still can't find a match, try a more general approach
                # Look for the mode number followed by any text and then yes/no
                general_pattern = rf"(?:C\.)?{mode}.*?(yes|no)"
                match = re.search(general_pattern, cleaned_response, re.IGNORECASE | re.DOTALL)
                
                if match:
                    value = 1 if match.group(1).lower() == 'yes' else 0
                    failure_modes[mode].append(value)
                else:
                    # If all attempts fail, default to 'no'
                    print(f"Warning: Could not find mode {mode} in response {i}")
                    failure_modes[mode].append(0)

    except Exception as e:
        raise ValueError(f"Error parsing response: {e}")
    return failure_modes


def parse_response_timeout(response: str) -> Dict[str, Any]:
    result = json.loads(response)
    failure_modes = result.get("failure_modes", {})

    expected = set(TIMEOUT_FAILURE_RUBRIC.keys())
    actual = set(failure_modes.keys())

    missing = expected - actual
    extra = actual - expected

    if missing:
        raise ValueError(f"Missing failure mode(s): {missing}")
    if extra:
        print(f"Warning: Unexpected failure mode(s): {extra}")

    return failure_modes


PARSE_RESPONSE_FUNCTIONS = {
    "base": parse_response_base,
    "mast": parse_response_mast,
    "timeout": parse_response_timeout
}

if __name__ == "__main__":

    trace = "Task: copy file A to B \n "
    
    prompt = MAKE_FAILURE_PROMPTS["mast"](trace)
    print(prompt)
