import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import torch.distributed as dist
import torch.multiprocessing as mp

RUNTIME_CFGS = [
    "quest",
    "hg",
]
DTYPE = torch.float16

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Llama-3.1-8B-Instruct", choices=["Llama-3.1-8B-Instruct"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("--task", type=str, help="task name", required=True)
    parser.add_argument("--method", choices=RUNTIME_CFGS, default="quest")
    parser.add_argument("--token_budget", type=int, default=None)
    parser.add_argument("--page_size", type=int, default=16)
    parser.add_argument("--topp", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_seq_len", type=str, default="1024", help="max sequence length for quest, can be a int or a list of int (for example: 512,1024,1024)")
    parser.add_argument("--max_seq_len_cpu", type=int, default=0)
    parser.add_argument("--max_kvmetadata_len", type=int, default=0)
    args = parser.parse_args(args)
    if "," in args.max_seq_len:
        args.max_seq_len = [int(length) for length in args.max_seq_len.split(",")]
    else:
        args.max_seq_len = int(args.max_seq_len)
        args.max_seq_len = [args.max_seq_len for _ in range(32)]
        # args.max_seq_len[0] = 32768
        # args.max_seq_len[1] = 32768
    return args

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name or "llama3" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name, out_path):
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        
        if args.method == "quest":
            model.quest_clear()
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                num_logits_to_keep=1,
            )[0]
        else:
            output = model.generate(
                input.input_ids,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                num_logits_to_keep=1,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device):
    # if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
    #     tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    #     model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    # elif "llama2" in model_name:
    #     replace_llama_attn_with_flash_attn()
    #     tokenizer = LlamaTokenizer.from_pretrained(path)
    #     model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    # elif "longchat" in model_name or "vicuna" in model_name:
    #     from fastchat.model import load_model
    #     replace_llama_attn_with_flash_attn()
    #     model, _ = load_model(
    #         path,
    #         device='cpu',
    #         num_gpus=0,
    #         load_8bit=False,
    #         cpu_offloading=False,
    #         debug=False,
    #     )
    #     model = model.to(device)
    #     model = model.bfloat16()
    #     tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    tokenizer = AutoTokenizer.from_pretrained(path)
    if args.method == "quest":
        from quest import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(path, device_map=device, torch_dtype=DTYPE)

        # Init Quest Controller
        model.quest_init(page_size=args.page_size, max_seq_len=args.max_seq_len, token_budget=args.token_budget, topp=args.topp, max_seq_len_cpu=args.max_seq_len_cpu, max_kvmetadata_len=args.max_kvmetadata_len)
    else:
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(path, device_map=device, torch_dtype=DTYPE)
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device(args.device)
    model_name = args.model
    # define your model1
    max_length = model2maxlen[model_name]
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = [args.task]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        if args.e:
            # data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            with open(f'/home/zhanghaoyu/datasets/LongBench/data/{dataset}_e.jsonl', 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f.readlines()]
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
            if args.method == "quest":
                out_path = f"pred_e/{model_name}/{dataset}_quest_tb{args.token_budget}_ps{args.page_size}"
                if args.topp is not None:
                    out_path += f"_topp{args.topp}"
                out_path += ".jsonl"
        else:
            # data = load_dataset('THUDM/LongBench', dataset, split='test')
            with open(f'/home/zhanghaoyu/datasets/LongBench/data/{dataset}.jsonl', 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f.readlines()]
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            out_path = f"pred/{model_name}/{dataset}.jsonl"
            if args.method == "quest":
                out_path = f"pred/{model_name}/{dataset}_quest_tb{args.token_budget}_ps{args.page_size}"
                if args.topp is not None:
                    out_path += f"_topp{args.topp}"
                out_path += ".jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        preds = get_pred(model, tokenizer, data_all, max_length, max_gen, prompt_format, dataset, device, model_name, out_path)
