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
    prompt: str, tokens: int, device: str, benchmark: Benchmark = Benchmark.OFF
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

    step_times = []
    ttft = None
    gen_start = time.perf_counter()

    with torch.no_grad():
        for i in range(tokens):
            input = torch.tensor(token_ids).unsqueeze(0).to(device)

            if benchmark != Benchmark.OFF:
                synchronize(device)
                step_start = time.perf_counter()
                logits = gpt(input)
                synchronize(device)
                step_time = time.perf_counter() - step_start
                if ttft is None:
                    ttft = step_time
                if benchmark == Benchmark.SYNCHRONIZE:
                    step_times.append((len(token_ids), step_time))
            else:
                logits = gpt(input)

            next_token = logits[0, -1].argmax().to("cpu").item()
            token_ids.append(next_token)
            next_token = tokenizer.decode([next_token])
            print(next_token, end="")
            result += next_token

            if i == 500:
                proc = psutil.Process(os.getpid())
                print("INFERENCE PROCESS MEMORY USAGE")
                print(proc.memory_info().rss / 1e9, "GB")

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
        print(f"TTFT:         {ttft * 1000:.1f}ms")
        print(f"Throughput:   {throughput:.2f} tokens/sec")
        print(f"Total time:   {total_time:.2f}s")
        if memory_gb is not None:
            print(f"Memory usage: {memory_gb:.2f} GB")
        if benchmark == Benchmark.SYNCHRONIZE:
            print(f"\nPer-step latency (seq_len, time_ms):")
            for seq_len, t in step_times:
                print(f"  {seq_len:5d}  {t * 1000:.1f}ms")

    return result


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
