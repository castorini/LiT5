sort_key=normswoquery

# Uncomment the first-stage you wish to test
#firststage=spladepp
firststage=bm25

# Uncomment the model you wish to test
model=castorini/LiT5-Score-base; batchsize=30
#model=castorini/LiT5-Score-large; batchsize=10
#model=castorini/LiT5-Score-xl; batchsize=4

for topics in 'dl19' 'dl20'; do
    runfile_path="runs/run.${topics}_${firststage}_${model//\//}.trec"

    python3 FiD/LiT5-Score.py \
        --model_path $model \
        --eval_data "topics/msmarco-${topics}-${firststage}.jsonl" \
        --batch_size $batchsize \
        --n_passages 100 \
        --runfile_path $runfile_path \
        --text_maxlength 150 \
        --answer_maxlength 20 \
        --write_crossattention_scores \
        --sort_key $sort_key \
        --bfloat16
        
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 ${topics}-passage $runfile_path
done 