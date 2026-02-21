import argparse
from inference import inference, Benchmark


def main():
    parser = argparse.ArgumentParser(
        prog="generate", description="Run GPT-2 and generate"
    )

    parser.add_argument("-p", "--prompt", type=str)
    parser.add_argument("-mt", "--max-tokens", type=int, default=1024)
    parser.add_argument("-d", "--device", type=str, default="mps")
    parser.add_argument(
        "-b",
        "--benchmark",
        type=str,
        choices=[b.value for b in Benchmark],
        default=Benchmark.OFF,
    )

    args = parser.parse_args()

    prompt = args.prompt
    if not prompt:
        print("No prompt provided")
        exit(1)
    max_tokens = args.max_tokens
    device = args.device
    benchmark = Benchmark(args.benchmark)

    print(
        f"Prompt: {prompt} | Max tokens: {max_tokens} | Device: {device} | Benchmark: {benchmark}"
    )
    inference(prompt, max_tokens, device, benchmark)


if __name__ == "__main__":
    main()
