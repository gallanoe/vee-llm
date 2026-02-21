from model import GPT2, Tokenizer
import torch
import time
from enum import StrEnum, auto

# Investigating weird memory usage
import psutil
import os


class Benchmark(StrEnum):
    SYNCHRONIZE = auto()
    TOTAL = auto()
    OFF = auto()


def inference(
    prompt: str,
    tokens: int,
    device: str,
    cache_enabled: bool = True,
    benchmark: Benchmark = Benchmark.OFF,
) -> str:
    load_start = time.perf_counter()
    gpt = GPT2.from_pretrained(device)
    gpt.eval()
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.2f}s")
    tokenizer = Tokenizer()

    print(prompt, end="")

    result = prompt
    token_ids = tokenizer.encode(prompt)

    ttft = None
    gen_start = time.perf_counter()

    # Prefill
    kv_cache = None
    if cache_enabled and benchmark == Benchmark.OFF:
        input = torch.tensor(token_ids[:-1]).unsqueeze(0).to(device)
        _, kv_cache = gpt(input)
    # TODO: Fix synchronize benchmark setting
    elif cache_enabled and benchmark != Benchmark.OFF:
        input = torch.tensor(token_ids[:-1]).unsqueeze(0).to(device)
        synchronize(device)
        _, kv_cache = gpt(input)
        synchronize(device)
    pos = len(token_ids) - 1

    with torch.no_grad():
        for i in range(tokens):
            if cache_enabled:
                input = torch.tensor([token_ids[-1]]).unsqueeze(0).to(device)
            else:
                input = torch.tensor(token_ids).unsqueeze(0).to(device)

            if benchmark != Benchmark.OFF:
                synchronize(device)
                step_start = time.perf_counter()
                logits, kv_cache = gpt(
                    input,
                    pos=torch.tensor([pos]).unsqueeze(0).to(device)
                    if cache_enabled
                    else None,
                    kv_cache=kv_cache if cache_enabled else None,
                )
                synchronize(device)
                step_time = time.perf_counter() - step_start
                if ttft is None:
                    ttft = step_time
                if benchmark == Benchmark.SYNCHRONIZE:
                    print(f"  [{len(token_ids):5d} tokens]  {step_time * 1000:.1f}ms")
            else:
                logits, kv_cache = gpt(
                    input,
                    pos=torch.tensor([pos]).unsqueeze(0).to(device)
                    if cache_enabled
                    else None,
                    kv_cache=kv_cache if cache_enabled else None,
                )

            next_token = logits[0, -1].argmax().to("cpu").item()
            token_ids.append(next_token)
            next_token = tokenizer.decode([next_token])
            pos += 1
            result += next_token

    if benchmark != Benchmark.OFF:
        total_time = time.perf_counter() - gen_start
        throughput = tokens / total_time

        if device == "mps":
            memory_gb = torch.mps.current_allocated_memory() / 1024**3
        elif device == "cuda":
            memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        else:
            memory_gb = None

        print(f"\n\n--- Benchmark ---")
        print(f"Model load:   {load_time:.2f}s")
        if ttft:
            print(f"TTFT:         {ttft * 1000:.1f}ms")
        print(f"Throughput:   {throughput:.2f} tokens/sec")
        print(f"Total time:   {total_time:.2f}s")
        if memory_gb is not None:
            print(f"Memory usage: {memory_gb:.2f} GB")

    return result


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
