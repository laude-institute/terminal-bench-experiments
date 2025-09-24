import os
import json
import re
from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import sys
from pathlib import Path
from anthropic import Anthropic
import openai

BASE_DIR = Path(__file__).parent.parent


sys.path.insert(0, str(BASE_DIR / "src"))
from trace_parser import TraceParser
from failure_classes import MAKE_FAILURE_PROMPTS, PARSE_RESPONSE_FUNCTIONS

load_dotenv()


class FailureProcessor:
    """
    Base class for processing failure traces with shared configuration and output format.

    Args:
        config: Dictionary containing configuration parameters
        max_trace_length: Maximum trace length to process


    All FailureProcessor subclasses must output failure modes according to OUTPUT_MODE_RUBRIC:

    the base class output is
    - category: string (failure classification category)
    - evidence: string (specific evidence from trace)  
    - confidence: float 0.0-1.0 (confidence score)
    - required_skill: string (skill needed to avoid this failure)
    """
    
    def __init__(self, config=None):
        """
        Initialize the failure processor with shared configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or {}
        self.max_trace_length = self.config.get("max_trace_length", None)
        
    def process_trace(self, trace: str, task_description: str = "", max_trace_length: int = None) -> Dict:
        """
        Process a failure trace and return standardized output.
        
        Args:
            trace: The trace text to process
            task_description: Optional description of what the task was supposed to do
            
        Returns:
            Standardized result dictionary with metadata
        """
        raise NotImplementedError("Subclasses must implement process_trace method")
    
    def _prepare_text(self, trace: str, task_description: str) -> str:
        """Prepare text which will be fed to the model.
        Here we combine the task description and the trace.
        """

        # Truncate trace if too long and max_trace_length is set
        if self.max_trace_length and len(trace) > self.max_trace_length:
            trace = trace[:self.max_trace_length] + "... [truncated]"
        
        # Combine task description and trace
        if task_description:
            text = f"Task: {task_description}\n\nExecution Trace:\n{trace}"
        else:
            text = f"Execution Trace:\n{trace}"
        
        return text
    


class FailureAnalyzerJudge(FailureProcessor):
    """Analyze failure traces using LLM classification as judge."""
    
    def __init__(self, config=None):
        """
        Initialize the failure analyzer.
        
        Args:
            config: Configuration dictionary with keys like 'llm_provider', 'model_name', etc.
        """
        super().__init__(config)
        self.model_name = self.config.get("model_name", None)  # Will be set based on provider
        self.llm_provider = self.config.get("llm_provider", self._get_model_provider())
        self.failure_prompt_type = self.config.get("failure_prompt_type", "mast")
        
        if self.llm_provider == "anthropic":
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model_name = self.config.get("model_name", "claude-3-haiku-20240307")
        elif self.llm_provider == "openai":
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.client = openai
            self.model_name = self.config.get("model_name", "gpt-4o-mini")
        elif self.llm_provider == "local":
            # Local model
            self.client = None
            self.model_name = self.config.get("model_name", "microsoft/deberta-v3-base")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _get_model_provider(self) -> str:
        """Get the model provider."""
        if self.model_name in ["gpt-4o", "gpt-4o-mini", "gpt-5", "gpt-5-nano"]:
            return "openai"
        elif self.model_name in ["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022", "claude-opus-4-1-20250805"]:
            return "anthropic"
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def _get_response(self, prompt : str) -> Dict[str, any]:
        """Get the response JSON from the LLM."""
        if self.llm_provider == "anthropic":
            response = self.client.messages.create(
                model=self.model_name,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        elif self.llm_provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=1.0,
                messages=[{"role": "user", "content": prompt}],
                timeout=300  # 5 minute timeout for GPT-5
            )
            return response.choices[0].message.content
        elif self.llm_provider == "local":
            return self._local_classify(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _parse_response(self, response: str) -> Dict[str, any]:
        """Parse the response JSON from the LLM."""
        parse_response_function = PARSE_RESPONSE_FUNCTIONS[self.failure_prompt_type]
        return parse_response_function(response)

        return failure_modes


    def process_trace(self, trace: str, task_description: str = "") -> List[Dict[str, any]]:
        """
        Classify a failure trace into one or more failure modes using LLM.
        
        Args:
            trace: The trace text to classify
            task_description: Optional description of what the task was supposed to do
            
        Returns:
            List of failure mode classifications with evidence and confidence
        """
        # Build the classification prompt
        prompt = self._prepare_text(trace, task_description)
        response_json = self._get_response(prompt)

        failure_modes = self._parse_response(response_json)
        return failure_modes

    
    def _prepare_text(self, trace: str, task_description: Optional[str] = None) -> str:
        """Build the prompt for LLM classification."""
        # Check trace length
        if self.max_trace_length and len(trace) > self.max_trace_length:
            strict_mode = self.config.get("strict_length_check", False)
            if strict_mode:
                raise ValueError(f"Execution trace is too long ({len(trace)} characters). "
                               f"Maximum allowed length is {self.max_trace_length} characters.")
        
        prompt = MAKE_FAILURE_PROMPTS[self.failure_prompt_type](trace, task_description)
        
        return prompt
    

class FailureEmbedder(FailureProcessor):
    """Generate embeddings for failure traces using specified embedding distance and model."""
    
    def __init__(self, config=None):
        """
        Initialize the failure embedder.
        
        Args:
            config: Configuration dictionary with keys like 'embedding_distance', 'model_name', etc.
        """
        super().__init__(config)
        self.embedding_distance = self.config.get("embedding_distance", "cosine")
        self.model_name = self.config.get("model_name", "text-embedding-3-small")
        
        # Initialize OpenAI client for embeddings
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def process_trace(self, trace: str, task_description: str = "") -> Dict:
        """
        Process a failure trace and return embedding results in standardized format.
        
        Args:
            trace: The trace text to embed
            task_description: Optional description of what the task was supposed to do
            
        Returns:
            Standardized result dictionary with embedding vector
        """
        embedding = self.embed_trace(trace, task_description)
        
        metadata = {
            "embedding_distance": self.embedding_distance,
            "model_name": self.model_name,
            "embedding_dimension": len(embedding),
            "trace_length": len(trace),
            "task_description": task_description
        }
        
        return self._create_standardized_output("embedding", embedding, metadata)
    
    def embed_trace(self, trace: str, task_description: str = "") -> List[float]:
        """
        Generate embeddings for a failure trace.
        
        Args:
            trace: The trace text to embed
            task_description: Optional description of what the task was supposed to do
            
        Returns:
            List of embedding values (floats)
        """
        # Prepare the text for embedding using parent method
        text_to_embed = self._prepare_text(trace, task_description)
        
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text_to_embed,
                encoding_format="float"
            )
            return response.data[0].embedding
                
        except Exception as e:
            print(f"Error generating embedding: {e}", flush=True)
            # Return zero vector on error
            return [0.0] * 1536  # Default size for text-embedding-3-small
    
    def compute_distance(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute distance between two embeddings using the specified distance metric.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Distance value (float)
        """
        import numpy as np
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        if self.embedding_distance == "cosine":
            # Cosine similarity, converted to distance
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            cosine_sim = dot_product / (norm1 * norm2)
            return 1 - cosine_sim
        elif self.embedding_distance == "euclidean":
            return np.linalg.norm(vec1 - vec2)
        elif self.embedding_distance == "manhattan":
            return np.sum(np.abs(vec1 - vec2))
        else:
            raise ValueError(f"Unsupported distance metric: {self.embedding_distance}")


FAILURE_CLASSIFIERS = {
    "judge": FailureAnalyzerJudge,
    "embedder": FailureEmbedder
}


def get_config(classifier_type: str, failure_prompt_type: str="mast", model_name: str=None):
    """Get the configuration for a classifier."""
    if classifier_type == "judge":
        config = {
            "model_name": model_name, 
            "failure_prompt_type": failure_prompt_type
        }
        return config
    elif classifier_type == "embedder":
        return {"embedding_distance": "cosine",
        "model_name": "text-embedding-3-small"}
    else:
        raise ValueError(f"Invalid classifier type: {classifier_type}")


if __name__ == "__main__":
    
    # Example usage
    parser = TraceParser(BASE_DIR / "traces")
    
    trial_id = "0d4d39d9-e67d-4610-8785-bab0ddbb2d81"
    failure_classifier = "judge"
    model_name="gpt-4o"
    # Read the trace
    trace = parser.parse_trace(trial_id)
    trace_text = trace.to_text(include_metadata=False)


    classifier = FAILURE_CLASSIFIERS[failure_classifier](config=get_config("judge", "mast", model_name))
    # Print classification prompt:
    prompt = classifier._prepare_text(trace_text, trace.get_task_name())

    print(prompt, flush=True)
    failure_modes = classifier.process_trace(trace_text, trace.get_task_name())

    print(failure_modes, flush=True)
    # Classify the trace
