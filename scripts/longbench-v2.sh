cd ./evaluation/LongBench

model="Llama-3.1-8B-Instruct"

python pred.py \
    --model $model \
    --method "hg"

for budget in 512 1024 2048 4096
do
    python pred.py \
        --model $model \
        --method "quest" \
        --token_budget $budget \
        --page_size 16
done

for topp in 0.1 0.2 0.3 0.4 0.5
do
    python pred.py \
        --model $model \
        --method "quest" \
        --token_budget 4096 \
        --page_size 16 \
        --topp $topp
done

python result.py