from transformers import AutoTokenizer
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import ceil

MODEL_PATH = "/home/zhanghaoyu/models/Llama-3.1-8B-Instruct/"
DEVICE = torch.device("cuda:0")
DTYPE = torch.float16
torch.set_default_dtype(DTYPE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

method = "hf"
token_budget = 512
topp = None
block_size = 16
topk = token_budget // block_size

if method == "quest":
  from quest import LlamaForCausalLM
  model = LlamaForCausalLM.from_pretrained(MODEL_PATH, device_map=DEVICE, torch_dtype=DTYPE, output_attentions=True)

  # Init Quest Controller
  model.quest_init(page_size=16, max_seq_len=8192, token_budget=token_budget, topp=topp)
else:
  from transformers import LlamaForCausalLM
  model = LlamaForCausalLM.from_pretrained(MODEL_PATH, device_map=DEVICE, torch_dtype=DTYPE, output_attentions=True)
  
prompt = "In an animal kingdom, the lion is the king. One day, the lion announces a competition to choose the most hardworking animal. The turtle, rabbit, monkey, zebra, and giraffe all decide to participate. After a day of observation, the lion notices that all the animals are working hard, except for the rabbit, who is sleeping. So why does the lion choose the rabbit as the most hardworking animal?"
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
print(f"Input Sequence Length: {inputs.input_ids.shape[1]}")

outputs = model.generate(
  **inputs,
  max_new_tokens=2000,
  output_attentions=True,
  return_dict_in_generate=True
)

generated_ids = outputs.sequences
attentions = outputs.attentions # (output_tokens, batch_size, num_heads, sequence_length, sequence_length)
all_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
all_tokens = [token.replace("Ġ", "") for token in all_tokens]
print(f"Generated Sequence Length: {len(all_tokens)}")
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

prompt_length = inputs.input_ids.shape[1]
# Search for the most similar attention matric to each attention matric
attentions = [list(step) for step in attentions]  # transfer tuple to list
attentions = [[layer.to(DEVICE) for layer in step] for step in attentions]  # transfer to cuda
num_layers = len(attentions[0])
num_heads = attentions[0][0].shape[1]
total_seq_len = len(all_tokens)
block_num = ceil(total_seq_len / block_size)
padded_seq_len = block_num * block_size
current_attention_matric = torch.zeros((total_seq_len, total_seq_len), device=DEVICE)
current_attention_matric_kvblock = torch.zeros((total_seq_len, block_num), device=DEVICE)
compare_attention_matric = torch.zeros((num_heads, total_seq_len, total_seq_len), device=DEVICE)
compare_attention_matric_kvblock = torch.zeros((num_heads, total_seq_len, block_num), device=DEVICE)
padded_attention = torch.zeros(total_seq_len, padded_seq_len, device='cuda')
mask = torch.zeros(total_seq_len, block_num, device='cuda')
for i in range(total_seq_len):
    max_block_idx = min(i // block_size + 1, block_num)  # i 对应的最大 block 索引
    mask[i, :max_block_idx] = 1
output = "Layer\tHead\tSimilar Layer\tSimilar Head\tDiff"
open('similar_block_attention_result.txt', 'w', encoding='utf-8').write(output + '\n')
for layer_i in range(1, num_layers):
    for head_i in range(num_heads):
        current_attention_matric.zero_()
        current_attention_matric[:attentions[0][layer_i][0, head_i].shape[0], :attentions[0][layer_i][0, head_i].shape[0]] = attentions[0][layer_i][0, head_i].detach()
        for i in range(1, len(attentions)):
            current_seq_len = prompt_length + i
            current_attention_matric[prompt_length + i - 1, :current_seq_len] = attentions[i][layer_i][0, head_i].detach()
        current_attention_matric_kvblock.zero_()
        padded_attention.zero_()
        padded_attention[:, :total_seq_len] = current_attention_matric
        reshaped_attention = padded_attention.view(total_seq_len, block_num, block_size)
        current_attention_matric_kvblock = torch.sum(reshaped_attention, dim=-1)
        _, indices = torch.topk(current_attention_matric_kvblock, k=topk, dim=1)
        # print("indices: ", indices)
        current_attention_matric_kvblock.zero_()
        current_attention_matric_kvblock.scatter_(1, indices, 1)
        # Change current_attention_matric_kvblock to the lower triangular matrix
        current_attention_matric_kvblock *= mask
        # print("current_attention_matric_kvblock: ", current_attention_matric_kvblock)
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(current_attention_matric_kvblock.cpu().numpy(), cmap="viridis")
        # plt.xlabel("Block Key Tokens")
        # plt.ylabel("Query Tokens")
        # plt.title(f"Block KV Attention Heatmap")
        # plt.show()
        # # save the attention matric figure
        # plt.savefig(f"block_attention_heatmap_layer_{layer_i}_head_{head_i}.png")
        # plt.close()
        similar_layer_head = (layer_i, head_i)
        min_diff = float("inf")
        for layer_j in range(layer_i):
            for head_j in range(len(attentions[0][layer_j][0])):
                compare_attention_matric[head_j, :attentions[0][layer_j][0, head_j].shape[0], :attentions[0][layer_j][0, head_j].shape[0]] = attentions[0][layer_j][0, head_j].detach()
                for i in range(1, len(attentions)):
                    current_seq_len = prompt_length + i
                    compare_attention_matric[head_j, prompt_length + i - 1, :current_seq_len] = attentions[i][layer_j][0, head_j].detach()
                compare_attention_matric_kvblock[head_j].zero_()
                padded_attention.zero_()
                padded_attention[:, :total_seq_len] = compare_attention_matric[head_j]
                reshaped_attention = padded_attention.view(total_seq_len, block_num, block_size)
                compare_attention_matric_kvblock[head_j] = torch.sum(reshaped_attention, dim=-1)
                _, indices = torch.topk(compare_attention_matric_kvblock[head_j], k=topk, dim=1)
                compare_attention_matric_kvblock[head_j].zero_()
                compare_attention_matric_kvblock[head_j].scatter_(1, indices, 1)
                # Change compare_attention_matric_kvblock to the lower triangular matrix
                compare_attention_matric_kvblock *= mask
                # plt.figure(figsize=(10, 8))
                # sns.heatmap(compare_attention_matric_kvblock[head_j].cpu().numpy(), cmap="viridis")
                # plt.xlabel("Block Key Tokens")
                # plt.ylabel("Query Tokens")
                # plt.title(f"Block KV Attention Heatmap")
                # plt.show()
                # # save the attention matric figure
                # plt.savefig(f"block_attention_heatmap_layer_{layer_j}_head_{head_j}.png")
                # plt.close()

            current_expanded = current_attention_matric_kvblock.unsqueeze(0).expand(num_heads, -1, -1)
            diffs = torch.sum(torch.abs(current_attention_matric_kvblock - compare_attention_matric_kvblock), dim=(1, 2)) # [num_heads]
            diff, min_index = torch.min(diffs, dim=0)
            if diff < min_diff:
                min_diff = diff
                similar_layer_head = (layer_j, min_index)
        print(f"Layer {layer_i}, Head {head_i} is most similar to Layer {similar_layer_head[0]}, Head {similar_layer_head[1]} with diff {min_diff}")
        output = f"{layer_i}\t{head_i}\t{similar_layer_head[0]}\t{similar_layer_head[1]}\t{min_diff}"
        open('similar_block_attention_result.txt', 'a', encoding='utf-8').write(output + '\n')
        ## plot the attention matric of the most similar layer and head
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(current_attention_matric, cmap="viridis")
        # plt.xlabel("Key Tokens")
        # plt.ylabel("Query Tokens")
        # plt.title(f"Attention Heatmap (Layer {layer_i}, Head {head_i})")
        # plt.show()
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(similar_attention_matric, cmap="viridis")
        # plt.xlabel("Key Tokens")
        # plt.ylabel("Query Tokens")
        # plt.title(f"Attention Heatmap (Layer {similar_layer_head[0]}, Head {similar_layer_head[1]})")
        # plt.show()

