# Based on Punica Project
# Check: https://github.com/efeslab/Atom/blob/main/e2e/punica-atom/benchmarks/bench_textgen.py

import argparse
import dataclasses
import time
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm.auto import tqdm
import os
import sys

schedule = torch.profiler.schedule(
    wait=128,
    warmup=64,
    active=16,
    repeat=1
)

@dataclasses.dataclass
class ModelConfig:
  model_path: str
  dtype: str = dataclasses.field(default="float16")
  device: str = dataclasses.field(default="cuda:0")

MODEL_CFGS = {
    "Llama-3.1-8B-Instruct":
        ModelConfig(
            model_path="/home/zhanghaoyu/models/Llama-3.1-8B-Instruct/"
        ),
}

def load_model(model_cfg: ModelConfig, args):
    device = torch.device(model_cfg.device)
    dtype = getattr(torch, model_cfg.dtype)
    torch.set_default_dtype(dtype)

    if args.method == "quest":
        from quest import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(model_cfg.model_path, device_map=device, torch_dtype=dtype)

        # Init Quest Controller
        max_seq_len = args.context_len + args.decode_len + 512
        model.quest_init(
            page_size=args.page_size,
            max_seq_len=max_seq_len,
            token_budget=args.token_budget,
            topp=args.topp,
            max_seq_len_cpu=args.max_seq_len_cpu,
            max_kvmetadata_len=args.max_kvmetadata_len,
            dtype=dtype,
            device=device
        )
    else:
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(model_cfg.model_path, device_map=device, torch_dtype=dtype)
    model = model.eval()
    return model

@torch.inference_mode()
def benchmark_quest():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CFGS.keys(), default="Llama-3.1-8B-Instruct")
    parser.add_argument("--method", type=str, default="quest")
    parser.add_argument("--context_len", type=int, default=2*1024)
    parser.add_argument("--decode_len", type=int, default=256)
    parser.add_argument("--page_size", type=int, default=16)
    parser.add_argument("--token_budget", type=int, default=256)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--topp", type=float, default=None)
    parser.add_argument("--max_seq_len", type=str, default="1024", help="max sequence length for quest, can be a int or a list of int (for example: 512,1024,1024)")
    parser.add_argument("--max_seq_len_cpu", type=int, default=0)
    parser.add_argument("--max_kvmetadata_len", type=int, default=0)
    args = parser.parse_args()

    assert args.model in MODEL_CFGS, f"Model {args.model} not found in MODEL_CFGS"
    model_cfg = MODEL_CFGS[args.model]
    
    max_seq_len = args.context_len + args.decode_len + 512
    page_size = args.page_size
    token_budget = args.token_budget
    context_len = args.context_len
    decode_len = args.decode_len

    model = load_model(model_cfg, args)
    
    dtype = getattr(torch, model_cfg.dtype)
    device = torch.device(model_cfg.device)
    hidden_size = model.config.hidden_size

    prefill_latency = []
    decode_latency = []

    for _ in tqdm(range(args.iteration)):
        # clear cuda cache
        torch.cuda.empty_cache()

        # Prefill Stage
        ts = time.perf_counter()
        hidden_states = torch.randn(1, context_len, hidden_size, dtype=dtype, device=device)
        model(
            inputs_embeds=hidden_states,
        )
        te = time.perf_counter()
        prefill_latency.append(te - ts)
        # Start decoding decode_len tokens
        with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], schedule=schedule, record_shapes=True, use_cuda=True) as prof:
            with record_function("model_inference"):
                for _ in range(decode_len):
                    ts = time.perf_counter()
                    hidden_states = torch.randn(1, 1, hidden_size, dtype=dtype, device=device)
                    model(
                        inputs_embeds=hidden_states,
                    )
                    te = time.perf_counter()
                    decode_latency.append(te - ts)
                    prof.step()
        output_file = "./result/{token_budget}-{page_size}-{context_len}-{decode_len}.json"
        if not os.path.exists("./result"):
            os.makedirs("./result")
        prof.export_chrome_trace(f"./result/{token_budget}-{page_size}-{context_len}-{decode_len}.json")
        if args.method == "quest":
            model.quest_clear()
    
    avg_prefill_latency = np.mean(prefill_latency)
    avg_decode_latency = np.mean(decode_latency)

    print("page_size,token_budget,context_len,decode_len,avg_prefill_latency,avg_decode_latency")
    print(f"{page_size},{token_budget},{context_len},{decode_len},{avg_prefill_latency},{avg_decode_latency}")

if __name__ == "__main__":
    benchmark_quest()

# nsys profile --delay 20 --duration 1 --output "$(env TZ='US/Pacific' date +%Y%m%d-%H%M%S).nsys-rep" python text_gen.py