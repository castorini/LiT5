# Uncomment the first-stage you wish to test
#firststage=spladepp
firststage=bm25

# Uncomment the model you wish to test
model=castorini/LiT5-Distill-base; batchsize=130
#model=castorini/LiT5-Distill-large; batchsize=65
#model=castorini/LiT5-Distill-xl; batchsize=18

total_n_rerank_passages=100
windowsize=20
stride=10

for topics in 'dl19' 'dl20'; do
    runfile_path="runs/run.${topics}_${firststage}_${model//\//}.trec"

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
    --bfloat16

    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 ${topics}-passage $runfile_path
done 