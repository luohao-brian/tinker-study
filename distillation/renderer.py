"""
Simplified Renderer for Qwen3 models.

Adapted from tinker-cookbook's renderers.py, containing only the essential
logic for Qwen3 chat template and supervised training.
"""

from enum import StrEnum
from typing import TypedDict, NotRequired

import tinker
import torch
from transformers import PreTrainedTokenizer


# Type definitions
Role = str
Content = str


class Message(TypedDict):
    """Container for a single turn in a conversation."""
    role: Role
    content: Content
    trainable: NotRequired[bool]


class TrainOnWhat(StrEnum):
    """Controls which tokens contribute to the training loss."""
    LAST_ASSISTANT_MESSAGE = "last_assistant_message"
    ALL_ASSISTANT_MESSAGES = "all_assistant_messages"
    ALL_MESSAGES = "all_messages"
    ALL_TOKENS = "all_tokens"


class Qwen3Renderer:
    """
    Renderer for Qwen3 models (ChatML format).
    
    Format:
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        Hello!<|im_end|>
        <|im_start|>assistant
        Hi there!<|im_end|>
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
    
    def get_stop_sequences(self) -> list[int]:
        """Return stop token IDs for sampling."""
        tokens = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        assert len(tokens) == 1, f"Expected single token for <|im_end|>, got {len(tokens)}"
        return tokens
    
    @property
    def _bos_tokens(self) -> list[int]:
        """Get BOS (beginning of sequence) tokens if any."""
        bos_token_str = self.tokenizer.bos_token
        if bos_token_str is None:
            return []
        assert isinstance(bos_token_str, str)
        return self.tokenizer.encode(bos_token_str, add_special_tokens=False)
    
    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        max_length: int | None = None,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        """
        Build tokens and weights for supervised training.
        
        Args:
            messages: List of conversation messages.
            train_on_what: Controls which tokens get loss weight.
            max_length: Optional max sequence length.
            
        Returns:
            Tuple of (ModelInput, weights_tensor).
        """
        chunks_weights: list[tuple[list[int], float]] = []
        
        # Add BOS token if present (with weight 0)
        if self._bos_tokens:
            chunks_weights.append((self._bos_tokens, 0.0))
        
        for idx, message in enumerate(messages):
            role = message["role"]
            content = message["content"]
            is_last = idx == len(messages) - 1
            is_assistant = role == "assistant"
            
            # Build prefix: <|im_start|>role\n (or \n<|im_start|>role\n for non-first)
            if idx == 0:
                prefix = f"<|im_start|>{role}\n"
            else:
                prefix = f"\n<|im_start|>{role}\n"
            
            # Build content with end token
            content_with_end = f"{content}<|im_end|>"
            
            # Tokenize
            prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
            content_tokens = self.tokenizer.encode(content_with_end, add_special_tokens=False)
            
            # Prefix always has weight 0 (unless ALL_TOKENS)
            prefix_weight = 1.0 if train_on_what == TrainOnWhat.ALL_TOKENS else 0.0
            chunks_weights.append((prefix_tokens, prefix_weight))
            
            # Content weight depends on train_on_what
            match train_on_what:
                case TrainOnWhat.LAST_ASSISTANT_MESSAGE:
                    content_weight = 1.0 if (is_last and is_assistant) else 0.0
                case TrainOnWhat.ALL_ASSISTANT_MESSAGES:
                    content_weight = 1.0 if is_assistant else 0.0
                case TrainOnWhat.ALL_MESSAGES:
                    content_weight = 1.0
                case TrainOnWhat.ALL_TOKENS:
                    content_weight = 1.0
                case _:
                    raise ValueError(f"Unknown train_on_what: {train_on_what}")
            
            chunks_weights.append((content_tokens, content_weight))
        
        # Flatten to tokens and weights
        all_tokens = []
        all_weights = []
        for tokens, weight in chunks_weights:
            all_tokens.extend(tokens)
            all_weights.extend([weight] * len(tokens))
        
        # Truncate if needed
        if max_length and len(all_tokens) > max_length:
            all_tokens = all_tokens[:max_length]
            all_weights = all_weights[:max_length]
        
        model_input = tinker.ModelInput.from_ints(all_tokens)
        weights_tensor = torch.tensor(all_weights)
        
        return model_input, weights_tensor


def get_renderer(renderer_name: str, tokenizer: PreTrainedTokenizer) -> Qwen3Renderer:
    """
    Get a renderer by name.
    
    Currently only supports 'qwen3', but can be extended.
    """
    if renderer_name.lower() in ("qwen3", "qwen"):
        return Qwen3Renderer(tokenizer)
    else:
        raise ValueError(f"Unknown renderer: {renderer_name}. Supported: qwen3")
