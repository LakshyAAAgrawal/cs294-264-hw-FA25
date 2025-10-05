from abc import ABC, abstractmethod
import os
from typing import List, Dict, Union

try:
    from vllm import LLM as VLLM
    from vllm import SamplingParams
except ImportError:
    pass


class LLM(ABC):
    """Abstract base class for Large Language Models."""

    @abstractmethod
    def generate(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        """
        Generate a response from the LLM given a prompt.
        
        Args:
            prompt: Either a string prompt or a list of OpenAI-formatted messages
                   with "role" and "content" keys.
        
        Returns:
            The generated text response.
        
        Must include any required stop-token logic at the caller level.
        """
        raise NotImplementedError


class OpenAIModel(LLM):
    """
    Example LLM implementation using OpenAI's Responses API.

    TODO(student): Implement this class to call your chosen backend (e.g., OpenAI GPT-5 mini)
    and return the model's text output. You should ensure the model produces the response
    format required by ResponseParser and include the stop token in the output string.
    """

    def __init__(self, stop_token: str, model_name: str = "gpt-5-mini", openai_model: bool = True):
        self.stop_token = stop_token
        self.model_name = model_name
        self.openai_model = openai_model

        if openai_model:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.client = VLLM(model_name)

    def generate(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        if self.openai_model:
            # Handle both string and list of messages formats
            # Note: OpenAI Responses API accepts list of message dicts as input
            response = self.client.responses.create(
                model=self.model_name, 
                tools=[{"type": "web_search_preview"}], 
                input=prompt  # type: ignore
            )

            # Get the text content and ensure stop token is present
            text = response.output_text
        else:
            # For VLLM, convert messages to string if needed
            if isinstance(prompt, list):
                # Convert message list to string format
                prompt_str = "\n\n".join([
                    f"{msg['role'].upper()}:\n{msg['content']}"
                    for msg in prompt
                ])
            else:
                prompt_str = prompt
            
            # VLLM client's generate method
            sampling_params = SamplingParams(temperature=0.1)
            outputs = self.client.generate(prompt_str, sampling_params=sampling_params)  # type: ignore
            text = outputs[0].outputs[0].text

        if text and not text.endswith(self.stop_token):
            text += self.stop_token

        return text