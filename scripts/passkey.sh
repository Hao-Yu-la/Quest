cd evaluation/passkey

MODEL=Llama-3.1-8B-Instruct
MODELPATH=/home/zhanghaoyu/models/Llama-3.1-8B-Instruct/
OUTPUT_DIR=results/$MODEL

mkdir -p $OUTPUT_DIR

length=25000

# for token_budget in 512 1024 2048 4096
# do
#     python passkey.py -m $MODEL --model_path $MODELPATH \
#         --iterations 100 --fixed-length $length \
#         --method quest --token_budget $token_budget --page_size 16 --max_seq_len 15000 \
#         --output-file $OUTPUT_DIR/$MODEL-quest-$token_budget.jsonl
# done

for topp in 0.1
do
    python passkey.py -m $MODEL --model_path $MODELPATH \
        --iterations 10 --fixed-length $length \
        --method quest --token_budget 4096 --topp $topp --page_size 16 --max_seq_len 8192 --max_seq_len_cpu 50512 \
        --output-file $OUTPUT_DIR/$MODEL-quest-$topp.jsonl
done
