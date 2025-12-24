import argparse
import asyncio
import json
import logging
import os
import time
from termcolor import colored
from pathlib import Path

import tinker
from transformers import AutoTokenizer

from renderer import Qwen3Renderer, TrainOnWhat


def load_conversations(file_path: str) -> list[dict]:
    """Load conversations from JSONL file."""
    conversations = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            if "messages" not in data:
                raise ValueError(
                    f"Each line must contain 'messages' field. Got: {data.keys()}"
                )
            conversations.append(data)
    return conversations


def datum_from_model_input_weights(
    model_input: tinker.ModelInput,
    weights: "torch.Tensor",
    max_length: int | None = None,
) -> tinker.Datum:
    # Get tokens from model input
    all_tokens = []
    for chunk in model_input.chunks:
        if hasattr(chunk, "tokens"):
            all_tokens.extend(chunk.tokens)
        else:
            # Image chunks would have a 'length' attribute
            all_tokens.extend([0] * chunk.length)

    # Truncate to max_length if needed
    if max_length is not None and len(all_tokens) > max_length:
        all_tokens = all_tokens[:max_length]
        weights = weights[:max_length]

    if len(all_tokens) < 2:
        raise ValueError("need at least 2 tokens for input/target split")

    # Right-shift inputs (remove last token) and left-shift targets (remove first token)
    input_tokens = all_tokens[:-1]
    target_tokens = all_tokens[1:]
    weights = weights[1:]  # Align weights with targets

    # Create model input from input tokens
    input_model_input = tinker.ModelInput.from_ints(input_tokens)

    return tinker.Datum(
        model_input=input_model_input,
        loss_fn_inputs={
            "weights": tinker.TensorData(
                data=weights.tolist(),
                dtype="float32",
                shape=list(weights.shape),
            ),
            "target_tokens": tinker.TensorData(
                data=target_tokens,
                dtype="int64",
                shape=[len(target_tokens)],
            ),
        },
    )

class SupervisedDataset:
    """Simple dataset for supervised training."""
    
    def __init__(
        self,
        conversations: list[dict],
        batch_size: int,
        renderer: Qwen3Renderer,
        max_length: int | None = None,
        train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    ):
        self.conversations = conversations
        self.batch_size = batch_size
        self.renderer = renderer
        self.max_length = max_length
        self.train_on_what = train_on_what
        self.shuffled_indices = list(range(len(conversations)))
    
    def get_batch(self, index: int) -> list[tinker.Datum]:
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.conversations))
        batch_indices = self.shuffled_indices[start:end]
        
        data = []
        for idx in batch_indices:
            conv = self.conversations[idx]
            model_input, weights = self.renderer.build_supervised_example(
                conv["messages"],
                train_on_what=self.train_on_what,
            )
            datum = datum_from_model_input_weights(
                model_input, weights, self.max_length
            )
            data.append(datum)
        return data
    
    def __len__(self) -> int:
        return len(self.conversations) // self.batch_size


def to_ints(chunk: tinker.ModelInputChunk, tokenizer: AutoTokenizer):
    if isinstance(chunk, tinker.EncodedTextChunk):
        return chunk.tokens
    else:
        (at_token,) = tokenizer.encode("@", add_special_tokens=False)
        return [at_token] * chunk.length


def format_colorized(
    tokens: list[int], weights: list[float], tokenizer: AutoTokenizer, draw_newline_arrow: bool = False
) -> str:
    """
    Colour-code text according to per-token weights.

    * Cyan text  → weight > 0
    * Yellow text  → weight = 0
    * Red text   → weight < 0

    The function minimises ANSI escape sequences by wrapping *runs* of
    like-coloured tokens, and decodes each run in a single call so that
    multi-byte or multibyte-character languages (e.g. CJK) render correctly.
    """
    if len(tokens) != len(weights):
        raise ValueError("`tokens` and `weights` must be the same length.")

    chunks, current_ids, current_color = [], [], None

    def flush_current_run():
        decoded = tokenizer.decode(current_ids)
        lines = decoded.splitlines(keepends=True)
        for line in lines:
            if draw_newline_arrow:
                line = line.replace("\n", "↵\n")
            chunks.append(colored(line, current_color))

    for tok_id, w in zip(tokens, weights, strict=True):
        if w < 0:
            color = "red"
        elif w == 0:
            color = "yellow"
        else:
            color = "green"

        # Flush when the colour changes
        if color != current_color and current_ids:
            flush_current_run()
            current_ids = []

        current_ids.append(tok_id)
        current_color = color

    flush_current_run()

    return "".join(chunks)


def colorize_example(datum: tinker.Datum, tokenizer: AutoTokenizer, key: str = "weights"):
    int_tokens = [
        token for chunk in datum.model_input.chunks for token in to_ints(chunk, tokenizer)
    ] + [datum.loss_fn_inputs["target_tokens"].tolist()[-1]]
    weights = [0.0] + datum.loss_fn_inputs[key].tolist()
    return format_colorized(int_tokens, weights, tokenizer)

if __name__ == "__main__":
    conversations = load_conversations("output/distillation_data.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-30B-A3B", use_fast=True, trust_remote_code=True
    )
    renderer = Qwen3Renderer(tokenizer)

    # Create dataset
    dataset = SupervisedDataset(
        conversations=conversations,
        batch_size=128,
        renderer=renderer,
        max_length=32768,
    )

    data = dataset.get_batch(10)
    print(len(data))
    print(data[0])
    print(colorize_example(data[0], tokenizer))
