import time
import argparse
import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification

# OPTIMIZATION 1: Force Single Thread (Crucial for low latency)
torch.set_num_threads(1)

def measure_latency(model_dir, input_file, runs=50):
    device = torch.device("cpu")
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
    
    # OPTIMIZATION 2: Dynamic Quantization
    print("Applying Dynamic Quantization...")
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    model.eval()

    print(f"Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        lines = f.readlines()
    texts = [json.loads(line)['text'] for line in lines if line.strip()][:100]

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for i in range(10):
            # OPTIMIZATION 3: Reduce max_length to 32
            inputs = tokenizer(texts[0], return_tensors="pt", padding=True, truncation=True, max_length=32).to(device)
            _ = model(**inputs)

    latencies = []
    print(f"Measuring latency over {runs} runs...")
    
    with torch.no_grad():
        for i in range(runs):
            text = texts[i % len(texts)]
            start_time = time.time()
            
            # Short Context Window (32 tokens is usually enough for STT commands)
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=32).to(device)
            outputs = model(**inputs)
            
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)

    print(f"\nLatency Stats (ms):")
    print(f"  p50: {p50:.2f} ms")
    print(f"  p95: {p95:.2f} ms")
    
    if p95 <= 20:
        print("✅ PASSED: p95 <= 20ms")
    else:
        print(f"❌ FAILED: p95 > 20ms (Current: {p95:.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()
    measure_latency(args.model_dir, args.input, args.runs)
