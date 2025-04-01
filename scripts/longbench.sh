cd ./evaluation/LongBench/LongBench

model="Llama-3.1-8B-Instruct"

for task in "qasper" "narrativeqa" "hotpotqa" "multifieldqa_en" "gov_report" "triviaqa"
do
    python -u pred.py \
        --model $model --task $task

    for budget in 512 1024 2048 4096
    do
        python -u pred.py \
            --model $model --task $task \
            --method "quest" --token_budget $budget --chunk_size 16
    done

    for topp in 0.1 0.2 0.3 0.4 0.5
    do
        python -u pred.py \
            --model $model --task $task \
            --method "quest" --token_budget 4096 --chunk_size 16 --topp $topp
    done
done

python -u eval.py --model $model