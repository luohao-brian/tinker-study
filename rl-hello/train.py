import argparse
import asyncio
import logging
import os
import random
import time
from typing import List, Tuple

import numpy as np
import tinker
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Configuration (aligned with agent-lightning hello.py) ---
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
NUM_TRAIN_TASKS = 1000
BATCH_SIZE = 32
SAMPLES_PER_PROMPT = 16
NUM_EPOCHS = 500
LEARNING_RATE = 1e-5
MAX_TOKENS = 32
SAVE_EVERY = 100

def parse_args():
    parser = argparse.ArgumentParser(description="Hello RL Training")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint name (e.g., 'batch_000020' or 'final')"
    )
    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="Starting batch number when resuming (for correct checkpoint naming)"
    )
    return parser.parse_args()

def load_env_simple():
    """
    Load TINKER_API_KEY from .env file or environment variable.
    Priority: environment variable > .env file
    """
    # Check if already set in environment
    if os.environ.get("TINKER_API_KEY"):
        logger.info("Using TINKER_API_KEY from environment variable")
        return
    
    # Try loading from .env file
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        logger.info(f"Loading environment variables from {env_path}")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    val = val.strip('\'"')
                    os.environ[key] = val
    else:
        logger.warning("No .env file found and TINKER_API_KEY not set in environment")

load_env_simple()

# --- 1. Renderer & Data Helpers ---

class SimpleQwenRenderer:
    """Minimal renderer for Qwen3 ChatML format."""
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.im_start = "<|im_start|>"
        self.im_end = "<|im_end|>"
        
    def build_prompt_str(self, task: str) -> str:
        """
        Constructs the prompt string for sampling.
        Matches agent-lightning hello.py:
        messages=[{"role": "user", "content": f"Let's play a game. Say you are {task}."}]
        """
        user_content = f"Let's play a game. Say you are {task}."
        prompt = (
            f"{self.im_start}user\n{user_content}{self.im_end}\n"
            f"{self.im_start}assistant\n"
        )
        return prompt

    def build_training_datum(
        self, 
        task: str, 
        response_text: str, 
        response_logprobs: List[float], 
        advantage: float
    ) -> tinker.Datum:
        """
        Constructs a Tinker Datum for importance_sampling.
        Per tinker-cookbook, requires: target_tokens, logprobs, advantages.
        (mask is removed before sending to server, so we don't include it)
        """
        prompt_str = self.build_prompt_str(task)
        
        # Tokenize (using tokenizer to get IDs)
        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        
        # Helper to construct response with im_end
        response_with_end = f"{response_text}{self.im_end}"
        response_ids = self.tokenizer.encode(response_with_end, add_special_tokens=False)
        
        # Ensure alignment between IDs and Logprobs
        min_len = min(len(response_ids), len(response_logprobs))
        if len(response_logprobs) > 0 and min_len < len(response_ids):
            response_ids = response_ids[:min_len]
        if len(response_logprobs) > 0 and min_len < len(response_logprobs):
            response_logprobs = response_logprobs[:min_len]

        all_ids = prompt_ids + response_ids
        
        # Inputs and Targets (per tinker-cookbook data_processing.py)
        # Input: tokens[:-1]
        # Target: tokens[1:]
        
        target_tokens = all_ids[1:]
        
        # Build logprobs and advantages for each target token
        # Prompt tokens get 0.0, response tokens get actual values
        full_logprobs = []
        full_advantages = []
        
        for i in range(len(target_tokens)):
            # target token index in all_ids is i+1
            token_idx = i + 1
            
            if token_idx < len(prompt_ids):
                # Target is part of prompt
                full_logprobs.append(0.0)
                full_advantages.append(0.0)
            else:
                # Target is part of response
                resp_idx = token_idx - len(prompt_ids)
                full_advantages.append(advantage)
                # response_logprobs might be shorter due to truncation
                if resp_idx < len(response_logprobs):
                    full_logprobs.append(response_logprobs[resp_idx])
                else:
                    full_logprobs.append(0.0) # Fallback

        model_input = tinker.ModelInput.from_ints(all_ids[:-1])
        
        # Per tinker-cookbook: send target_tokens, logprobs, advantages
        # (mask is removed before API call in their code, so we don't include it)
        return tinker.Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_torch(
                    torch.tensor(target_tokens, dtype=torch.int64)
                ),
                "logprobs": tinker.TensorData.from_torch(
                    torch.tensor(full_logprobs, dtype=torch.float32)
                ),
                "advantages": tinker.TensorData.from_torch(
                    torch.tensor(full_advantages, dtype=torch.float32)
                ),
            },
        )

# --- Helper Functions ---

async def save_checkpoint_async(
    training_client: tinker.TrainingClient,
    name: str,
    kind: str = "both",  # "state", "sampler", or "both"
) -> dict:
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

# --- 2. Logic & Main Loop ---

def compute_reward(task: str, response_content: str) -> float:
    """Exact reward logic."""
    content_lower = response_content.lower() if response_content else ""
    if ("i am " + task) in content_lower or ("i'm " + task) in content_lower:
        return 1.0
    elif ("not " + task) in content_lower:
        return -1.0
    else:
        return 0.0

async def main():
    args = parse_args()
    
    logger.info("Starting Hello Tinker RL (Importance Sampling)...")
    
    # 1. Setup
    logger.info(f"Loading Tokenizer {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
    renderer = SimpleQwenRenderer(tokenizer)
    
    logger.info("Connecting to Tinker Service...")
    service_client = tinker.ServiceClient()
    
    trainer_fp = await service_client.create_lora_training_client_async(
        base_model=MODEL_NAME,
        rank=32,
    )
    logger.info("Training Client Created.")
    
    # Get sampling client (resume from checkpoint or start fresh)
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        sampler = trainer_fp.create_sampling_client(args.resume)
        global_batch = args.start_batch
        logger.info(f"Starting from batch {global_batch}")
    else:
        logger.info("Creating initial Sampling Client...")
        sampler = await trainer_fp.save_weights_and_get_sampling_client_async("init")
        global_batch = 0
    logger.info("Sampling Client Ready.")

    # 2. Training Loop
    for epoch in range(NUM_EPOCHS):
        logger.info(f"=== Epoch {epoch + 1}/{NUM_EPOCHS} ===")
        
        tasks = [str(i) for i in range(NUM_TRAIN_TASKS)]
        random.shuffle(tasks)
        
        epoch_rewards = []
        
        for i in range(0, len(tasks), BATCH_SIZE):
            batch_tasks = tasks[i : i + BATCH_SIZE]
            
            # 1. Build Prompts
            prompts = [renderer.build_prompt_str(t) for t in batch_tasks]
            
            # 2. Sample
            async def sample_one(p):
                # Sample multiple times per prompt (group_size)
                return await sampler.sample_async(
                    prompt=tinker.ModelInput.from_ints(tokenizer.encode(p, add_special_tokens=False)),
                    num_samples=SAMPLES_PER_PROMPT,
                    sampling_params=tinker.SamplingParams(
                        max_tokens=MAX_TOKENS,
                        stop=tokenizer.encode("<|im_end|>", add_special_tokens=False),
                        temperature=1.0,
                    )
                )
            
            sample_futures = [sample_one(p) for p in prompts]
            sample_results = await asyncio.gather(*sample_futures)
            
            # 3. Process Results & Calculate Rewards
            batch_data_info = [] 
            batch_rewards_raw = []
            
            for task, result in zip(batch_tasks, sample_results):
                # Each result has multiple sequences (SAMPLES_PER_PROMPT)
                for seq in result.sequences:
                    response_text = ""
                    response_logprobs = [] # default empty
                    
                    if hasattr(seq, 'logprobs') and seq.logprobs:
                        response_logprobs = seq.logprobs
                    
                    # Reconstruct text from tokens
                    if hasattr(seq, 'tokens') and seq.tokens:
                        response_text = tokenizer.decode(seq.tokens)
                    elif hasattr(seq, 'chunks'):
                         for chunk in seq.chunks:
                            if isinstance(chunk, tinker.EncodedTextChunk):
                                response_text += chunk.text
                    
                    reward = compute_reward(task, response_text)
                    batch_rewards_raw.append(reward)
                    
                    # Print prompt and completion for debugging
                    reward_symbol = "✓" if reward == 1.0 else ("✗" if reward == -1.0 else "○")
                    logger.info(f"  [{reward_symbol}] Task: {task} | Response: {response_text.strip()[:50]}... | Reward: {reward}")
                    
                    batch_data_info.append({
                        "task": task,
                        "text": response_text,
                        "logprobs": response_logprobs,
                        "reward": reward
                    })
            
            epoch_rewards.extend(batch_rewards_raw)
            
            # 4. Compute Advantages (Center rewards)
            rewards_np = np.array(batch_rewards_raw)
            if len(rewards_np) > 1 and np.std(rewards_np) > 1e-6:
                advantages = rewards_np - np.mean(rewards_np)
            else:
                advantages = np.zeros_like(rewards_np)
            
            # 5. Build Training Data
            train_data = []
            for info, adv in zip(batch_data_info, advantages):
                datum = renderer.build_training_datum(
                    info["task"], 
                    info["text"], 
                    info["logprobs"], 
                    float(adv)
                )
                train_data.append(datum)
            
            # 6. Train Step
            if train_data:
                output = await trainer_fp.forward_backward_async(
                    data=train_data,
                    loss_fn="importance_sampling"
                )
                await output.result_async()
                
                optim_future = await trainer_fp.optim_step_async(
                    tinker.AdamParams(learning_rate=LEARNING_RATE)
                )
                await optim_future.result_async()
            
            global_batch += 1
            logger.info(f"Batch {global_batch}: Avg Reward = {np.mean(batch_rewards_raw):.2f}")
            
            # 7. Update sampler to use latest weights (every batch)
            sampler = await trainer_fp.save_weights_and_get_sampling_client_async(f"batch_{global_batch}")
            
            # 8. Save persistent checkpoint every SAVE_EVERY batches
            if global_batch % SAVE_EVERY == 0:
                checkpoint_name = f"batch_{global_batch:06d}"
                await save_checkpoint_async(trainer_fp, checkpoint_name, kind="sampler")

        avg_reward = np.mean(epoch_rewards)
        logger.info(f"Epoch {epoch+1} Complete. Average Reward: {avg_reward:.4f}")

    # Save final checkpoint
    logger.info("Training complete. Saving final checkpoint...")
    await save_checkpoint_async(trainer_fp, "final", kind="sampler")

if __name__ == "__main__":
    asyncio.run(main())
