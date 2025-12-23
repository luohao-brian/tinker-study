"""
Prompt Distillation - Create Data

This script uses a teacher model to generate language classification responses
for multilingual text samples. The output is saved as JSONL for SFT training.

Usage:
    python create_data.py --output_file ./output/distillation_data.jsonl

Environment:
    TINKER_API_KEY: Your Tinker API key
"""

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path

import tinker
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer

from renderer import Qwen3Renderer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


LANGUAGE_CLASSIFICATION_PROMPT = """You are a precise language classifier.

Goal: Classify the language of the provided text into exactly one of these labels:
ar (Arabic), de (German), el (Greek), en (English), es (Spanish), fr (French),
hi (Hindi), ru (Russian), tr (Turkish), ur (Urdu), vi (Vietnamese),
zh (Chinese - Simplified), ot (Other/Unknown).

Instructions:
1) Preprocess carefully (without changing the intended meaning):
   - Trim whitespace.
   - Ignore URLs, emails, file paths, hashtags, user handles, and emojis.
   - Ignore numbers, math expressions, and standalone punctuation.
   - If there is code, IGNORE code syntax (keywords, operators, braces) and focus ONLY on human language in comments and string literals.
   - Preserve letters and diacritics; do NOT strip accents.
   - If after ignoring the above there are no alphabetic letters left, output 'ot'.

2) Script-based rules (highest priority):
   - Devanagari script → hi.
   - Greek script → el.
   - Cyrillic script → ru.
   - Han characters (中文) → zh. (Treat Traditional as zh too.)
   - Arabic script → ar vs ur:
       • If Urdu-only letters appear (e.g., ے, ڑ, ں, ھ, ٹ, ڈ, کھ, گ, چ with Urdu forms), or clear Urdu words, choose ur.
       • Otherwise choose ar.
   (If multiple scripts appear, pick the script that contributes the majority of alphabetic characters. If tied, go to step 5.)

3) Latin-script heuristics (use when text is mainly Latin letters):
   - vi: presence of Vietnamese-specific letters/diacritics (ă â ê ô ơ ư đ, plus dense diacritics across many words).
   - tr: presence of Turkish-specific letters (ı İ ğ Ğ ş Ş ç Ç ö Ö ü Ü) and common function words (ve, bir, için, değil, ama, çok).
   - de: presence of umlauts (ä ö ü) or ß and common function words (und, der, die, das, nicht, ist).
   - es: presence of ñ, ¿, ¡ and common words (y, de, la, el, es, no, por, para, con, gracias, hola).
   - fr: frequent French diacritics (é è ê à ç ô â î û ù) and common words (et, le, la, les, des, une, est, avec, pour, merci, bonjour).
   - en: default among Latin languages if strong evidence for others is absent, but ONLY if English function words are present (the, and, is, are, to, of, in, for, on, with). If evidence is insufficient for any Latin language, prefer 'ot' over guessing.

4) Named entities & loanwords:
   - Do NOT decide based on a single proper noun, brand, or place name.
   - Require at least two function words or repeated language-specific signals (diacritics/letters) before assigning a Latin-language label.

5) Mixed-language text:
   - Determine the dominant language by counting indicative tokens (language-specific letters/diacritics/function words) AFTER preprocessing.
   - If two or more languages are equally dominant or the text is a deliberate multi-language mix, return 'ot'.

6) Very short or noisy inputs:
   - If the text is ≤2 meaningful words or too short to be confident, return 'ot' unless there is a very strong language-specific signal (e.g., "bonjour" → fr, "hola" → es).

7) Transliteration/romanization:
   - If Hindi/Urdu/Arabic/Chinese/Russian/Greek is written purely in Latin letters (romanized) without clear, repeated language-specific cue words, return 'ot'. (Only classify as hi/ur/ar/zh/ru/el when native scripts or highly distinctive romanized patterns are clearly present.)

8) Code-heavy inputs:
   - If the text is mostly code with minimal or no natural-language comments/strings, return 'ot'.
   - If comments/strings clearly indicate a language per rules above, use that label.

9) Ambiguity & confidence:
   - When in doubt, choose 'ot' rather than guessing.

Output format:
- Respond with EXACTLY one line: "Final Answer: xx"
- Where xx ∈ {{ar, de, el, en, es, fr, hi, ru, tr, ur, vi, zh, ot}} and nothing else.

Text to classify:
{text}
"""

VALID_LABELS = {"ar", "de", "el", "en", "es", "fr", "hi", "ru", "tr", "ur", "vi", "zh", "ot"}


def parse_args():
    parser = argparse.ArgumentParser(description="Create prompt distillation data")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-30B-A3B",
        help="Teacher model name",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to input multilingual text file (one sentence per line)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./output/distillation_data.jsonl",
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.15,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000,
        help="Maximum tokens to generate",
    )
    return parser.parse_args()


async def create_data_async(
    sentences: list[str],
    sampling_client,
    tokenizer,
    renderer: Qwen3Renderer,
    temperature: float,
    max_tokens: int,
) -> list[tuple[str, str | None]]:
    """Generate language classifications for all sentences using the teacher model."""

    async def sample_from_model(sentence: str) -> tuple[str, str | None]:
        prompt = LANGUAGE_CLASSIFICATION_PROMPT.format(text=sentence)
        tokenized_prompt = tinker.ModelInput.from_ints(tokenizer.encode(prompt))
        params = tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=renderer.get_stop_sequences(),  # Use renderer's stop sequences
        )
        result = await sampling_client.sample_async(
            prompt=tokenized_prompt,
            sampling_params=params,
            num_samples=1,
        )
        response = tokenizer.decode(result.sequences[0].tokens)
        
        # Parse "Final Answer: xx" from response
        search_response = re.search(r"Final Answer:\s*(\w+)", response)
        if search_response:
            answer = search_response.group(1).lower()
            if answer in VALID_LABELS:
                return (sentence, answer)
        return (sentence, None)

    results = []
    for coro in tqdm_asyncio.as_completed(
        [sample_from_model(s) for s in sentences],
        total=len(sentences),
        desc="Generating data",
    ):
        question, answer = await coro
        results.append((question, answer))

    return results


def main():
    args = parse_args()

    # Disable tokenizer parallelism warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Determine input file
    if args.input_file:
        input_file = args.input_file
    else:
        # Use default sample data
        input_file = Path(__file__).parent / "data" / "multilingual.txt"

    # Load sentences
    logger.info(f"Loading sentences from: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(sentences)} sentences")

    # Check output file - skip if exists (matching tinker-cookbook behavior)
    output_file = Path(args.output_file)
    if output_file.exists():
        logger.info(f"Output file {output_file} already exists, skipping.")
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Create tinker clients
    logger.info("Creating Tinker service client...")
    service_client = tinker.ServiceClient()
    
    logger.info(f"Creating sampling client for model: {args.model}")
    sampling_client = service_client.create_sampling_client(base_model=args.model)
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    
    # Create renderer for Qwen3
    renderer = Qwen3Renderer(tokenizer)

    # Generate data
    logger.info("Starting data generation...")
    results = asyncio.run(
        create_data_async(
            sentences=sentences,
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            renderer=renderer,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    )

    # Save results
    valid_count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for question, answer in results:
            if answer is not None:
                messages = {
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ]
                }
                f.write(json.dumps(messages, ensure_ascii=False) + "\n")
                valid_count += 1

    logger.info("Data generation complete!")
    logger.info(f"Valid samples: {valid_count}/{len(results)}")
    logger.info(f"Output saved to: {output_file}")


if __name__ == "__main__":
    main()
