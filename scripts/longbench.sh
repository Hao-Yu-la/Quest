cd ./evaluation/LongBench/LongBench

model="Llama-3.1-8B-Instruct"

for task in "qasper" "narrativeqa" # "triviaqa" "hotpotqa" "qasper" "narrativeqa" "multifieldqa_en" "gov_report" 
do
    # for budget in 512 1024 2048 4096 
    # do
    #     python -u pred.py \
    #         --model $model --task $task \
    #         --method "quest" --token_budget $budget --page_size 16 --device "cuda:0" --max_seq_len 16896
    # done

    for topp in 0.001 0.01 0.05
    do
        python -u pred.py \
            --model $model --task $task \
            --method "quest" --token_budget 4096 --page_size 16 --topp $topp --device "cuda:0" --max_seq_len 16896
    done
    
    # python -u pred.py \
    #     --model $model --task $task --method "hg" --device "cuda:0"

done

python -u eval.py --model $model