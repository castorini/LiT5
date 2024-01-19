# Uncomment the first-stage you wish to test
#firststage=spladepp
firststage=bm25

# Uncomment the model you wish to test
model=castorini/LiT5-Distill-base; batchsize=260
#model=castorini/LiT5-Distill-large; batchsize=120
#model=castorini/LiT5-Distill-xl; batchsize=36

total_n_rerank_passages=100
windowsize=20
stride=10
n_passes=1

for topics in 'dl19' 'dl20' 'dl21' 'dl22'; do
    runfile_path="runs/run.${topics}_${firststage}_${model//\//}"

    python3 FiD/LiT5-Distill.py \
    --model_path $model \
    --eval_data "topics/msmarco-${topics}-${firststage}.jsonl" \
    --batch_size $batchsize \
    --n_passages $windowsize \
    --runfile_path $runfile_path \
    --text_maxlength 150 \
    --answer_maxlength 100 \
    --stride $stride \
    --n_rerank_passages $total_n_rerank_passages \
    --bfloat16 \
    --n_passes $n_passes

    for ((i = 0 ; i < n_passes ; i++ )); do
        python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 ${topics}-passage ${runfile_path}.${i}.trec
    done
done 