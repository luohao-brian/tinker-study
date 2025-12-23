"""
Prompt Distillation - SFT Training

This script fine-tunes a student model on the distilled data using Tinker's
training API with LoRA.

Usage:
    python sft.py --data_file ./output/distillation_data.jsonl

Environment:
    TINKER_API_KEY: Your Tinker API key
"""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

import tinker
from transformers import AutoTokenizer

from renderer import Qwen3Renderer, TrainOnWhat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SFT training for prompt distillation")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-30B-A3B",
        help="Model name for training",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="./output/distillation_data.jsonl",
        help="Path to training data JSONL file",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=4,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help="LoRA rank",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=32768,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=20,
        help="Save checkpoint every N steps (0 to disable)",
    )
    return parser.parse_args()


def load_conversations(file_path: str) -> list[dict]:
    """Load conversations from JSONL file."""
    conversations = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            if "messages" not in data:
                raise ValueError(f"Each line must contain 'messages' field. Got: {data.keys()}")
            conversations.append(data)
    return conversations


def datum_from_model_input_weights(
    model_input: tinker.ModelInput,
    weights: "torch.Tensor",
    max_length: int | None = None,
) -> tinker.Datum:
    """
    Create a Datum from a ModelInput and weights tensor.
    
    Adapted from tinker-cookbook's common.py. Performs:
    - Max length truncation
    - Right-shifting inputs and left-shifting targets for next-token prediction
    
    Args:
        model_input: The model input containing tokens
        weights: The weights tensor aligned with the model_input length
        max_length: Optional maximum sequence length
        
    Returns:
        A Datum with model_input (input tokens) and loss_fn_inputs (target tokens and weights)
    """
    # Get tokens from model input
    all_tokens = []
    for chunk in model_input.chunks:
        if hasattr(chunk, 'tokens'):
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


def compute_mean_nll(
    logprobs_list: list,
    weights_list: list,
) -> float:
    """
    Compute weighted mean negative log likelihood.
    
    Adapted from tinker-cookbook's common.py.
    
    Args:
        logprobs_list: List of logprobs TensorData from loss_fn_outputs.
        weights_list: List of weights TensorData from datum.loss_fn_inputs.
    
    Returns:
        The weighted mean NLL value.
    """
    import torch
    
    total_weighted_logprobs = 0.0
    total_weights = 0.0
    
    for logprobs, weights in zip(logprobs_list, weights_list, strict=True):
        # TensorData can be converted to torch
        if hasattr(logprobs, 'to_torch'):
            logprobs_t = logprobs.to_torch()
        else:
            logprobs_t = torch.tensor(logprobs.data if hasattr(logprobs, 'data') else logprobs)
        
        if hasattr(weights, 'to_torch'):
            weights_t = weights.to_torch()
        else:
            weights_t = torch.tensor(weights.data if hasattr(weights, 'data') else weights)
        
        total_weighted_logprobs += float(logprobs_t.dot(weights_t))
        total_weights += float(weights_t.sum())
    
    if total_weights == 0:
        return float("nan")
    
    return float(-total_weighted_logprobs / total_weights)


async def save_checkpoint_async(
    training_client: tinker.TrainingClient,
    name: str,
    kind: str = "both",  # "state", "sampler", or "both"
) -> dict[str, str]:
    """
    Save model checkpoint.
    
    Adapted from tinker-cookbook's checkpoint_utils.py.
    
    Args:
        training_client: Training client to save from
        name: Name for the checkpoint
        kind: What to save - "state" (training), "sampler" (inference), or "both"
    
    Returns:
        Dict with saved paths (state_path and/or sampler_path)
    """
    futures = {}
    if kind in ["state", "both"]:
        futures["state"] = await training_client.save_state_async(name)
    if kind in ["sampler", "both"]:
        futures["sampler"] = await training_client.save_weights_for_sampler_async(name)
    
    results = {k: await v.result_async() for k, v in futures.items()}
    paths = {k + "_path": v.path for k, v in results.items()}
    
    logger.info(f"Saved checkpoint '{name}': {paths}")
    
    return paths


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
    
    def set_epoch(self, seed: int = 0):
        """Shuffle the dataset for a new epoch."""
        import random
        random.seed(seed)
        self.shuffled_indices = list(range(len(self.conversations)))
        random.shuffle(self.shuffled_indices)
    
    def __len__(self) -> int:
        return len(self.conversations) // self.batch_size


async def train_async(args):
    """Main async training loop using Tinker."""
    
    # Load data
    logger.info(f"Loading data from: {args.data_file}")
    conversations = load_conversations(args.data_file)
    logger.info(f"Loaded {len(conversations)} conversations")
    
    # Load tokenizer and create renderer
    logger.info(f"Loading tokenizer for: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    renderer = Qwen3Renderer(tokenizer)
    
    # Create dataset
    dataset = SupervisedDataset(
        conversations=conversations,
        batch_size=args.batch_size,
        renderer=renderer,
        max_length=args.max_length,
    )
    n_batches = len(dataset)
    total_steps = n_batches * args.num_epochs
    
    logger.info(f"Training: {n_batches} batches x {args.num_epochs} epochs = {total_steps} steps")
    
    # Create Tinker training client
    logger.info("Creating Tinker service client...")
    service_client = tinker.ServiceClient()
    
    logger.info(f"Creating LoRA training client (rank={args.lora_rank})...")
    training_client = await service_client.create_lora_training_client_async(
        base_model=args.model_name,
        rank=args.lora_rank,
    )
    
    # Training loop with async pipeline
    # Submit next batch before waiting for current batch results
    logger.info("Starting training...")
    start_time = time.time()
    
    # Track pending batch for pipelining
    pending_batch = None
    
    async def submit_batch(epoch_idx: int, batch_idx: int):
        """Submit a batch for training, returns futures and metadata."""
        step = epoch_idx * n_batches + batch_idx
        batch_start = time.time()
        
        # Get batch data
        data = dataset.get_batch(batch_idx)
        if not data:
            return None
        
        # Compute learning rate (linear decay)
        progress = step / max(total_steps, 1)
        lr = args.learning_rate * (1.0 - progress)
        
        adam_params = tinker.AdamParams(
            learning_rate=lr,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )
        
        # Submit forward-backward and optimizer step (don't wait yet)
        fwd_bwd_future = await training_client.forward_backward_async(
            data, loss_fn="cross_entropy"
        )
        optim_step_future = await training_client.optim_step_async(adam_params)
        
        return {
            "step": step,
            "epoch_idx": epoch_idx,
            "batch_idx": batch_idx,
            "batch_start": batch_start,
            "lr": lr,
            "data": data,
            "fwd_bwd_future": fwd_bwd_future,
            "optim_step_future": optim_step_future,
        }
    
    async def finish_batch(batch_info: dict):
        """Wait for batch to complete and log metrics."""
        step = batch_info["step"]
        
        # Wait for results
        fwd_bwd_result = await batch_info["fwd_bwd_future"].result_async()
        await batch_info["optim_step_future"].result_async()
        
        # Compute metrics - matching cookbook's approach:
        # logprobs come from loss_fn_outputs, weights come from submitted data
        logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        weights = [datum.loss_fn_inputs["weights"] for datum in batch_info["data"]]
        mean_nll = compute_mean_nll(logprobs, weights)
        
        batch_time = time.time() - batch_info["batch_start"]
        
        # Log progress
        if step % 10 == 0 or step == total_steps - 1:
            logger.info(
                f"Step {step}/{total_steps} | "
                f"Epoch {batch_info['epoch_idx'] + 1}/{args.num_epochs} | "
                f"NLL: {mean_nll:.4f} | "
                f"LR: {batch_info['lr']:.2e} | "
                f"Time: {batch_time:.2f}s"
            )
        
        # Save checkpoint
        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            checkpoint_name = f"step_{step:06d}"
            await save_checkpoint_async(
                training_client=training_client,
                name=checkpoint_name,
                kind="both",
            )
    
    # Training loop with async pipelining
    # 
    # Pipelining pattern: submit batch N+1 before waiting for batch N to finish.
    # This keeps the GPU busy and improves throughput:
    #   - While GPU processes batch N, CPU prepares batch N+1
    #   - Reduces idle time between batches
    #
    # Timeline:
    #   Batch 0:  [Submit]─────[GPU计算]─────
    #   Batch 1:         [Submit]─────[GPU计算]─────
    #   Batch 2:                 [Submit]─────[GPU计算]─────
    #
    for epoch_idx in range(args.num_epochs):
        logger.info(f"Starting epoch {epoch_idx + 1}/{args.num_epochs}")
        dataset.set_epoch(seed=epoch_idx)
        
        for batch_idx in range(n_batches):
            # Step 1: Submit current batch to GPU (non-blocking)
            submitted = await submit_batch(epoch_idx, batch_idx)
            
            # Step 2: Wait for previous batch to finish 
            # (overlap: GPU works on pending while we submitted current)
            if pending_batch is not None:
                await finish_batch(pending_batch)
            
            # Step 3: Current batch becomes pending for next iteration
            pending_batch = submitted
    
    # Finish the last pending batch
    if pending_batch is not None:
        await finish_batch(pending_batch)
    
    # Save final checkpoint
    await save_checkpoint_async(
        training_client=training_client,
        name="final",
        kind="both",
    )
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f}s")


def main():
    args = parse_args()
    
    # Disable tokenizer parallelism warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Check data file exists
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")
    
    # Run training
    asyncio.run(train_async(args))


if __name__ == "__main__":
    main()
