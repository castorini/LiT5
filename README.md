# LiT5 (List-in-T5) Reranking

## 📟 Instructions

We provide the scripts and data necessary to reproduce reranking results for [LiT5-Distill](LiT5-Distill.sh) and [LiT5-Score](LiT5-Score.sh) on DL19 and DL20 for BM25 and SPLADE++ ED first-stage retrieval. Note you may need to change the batchsize depending on your VRAM. We have observed that results may change slightly when the batchsize is changed. [This is a known issue when running inference in bfloat16](https://github.com/huggingface/transformers/issues/25921). Additionally, you may need to remove the --bfloat16 option from the scripts if your GPU does not support it.

Note, the v2 LiT5-Distill models support reranking up to 100 passages at once.

## Models

The following is a table of our models hosted on HuggingFace:

| Model Name            | Hugging Face Identifier/Link                                                               |
|-----------------------|--------------------------------------------------------------------------------------------|
| LiT5-Distill-base     | [castorini/LiT5-Distill-base](https://huggingface.co/castorini/LiT5-Distill-base)          |
| LiT5-Distill-large    | [castorini/LiT5-Distill-large](https://huggingface.co/castorini/LiT5-Distill-large)        |
| LiT5-Distill-xl       | [castorini/LiT5-Distill-xl](https://huggingface.co/castorini/LiT5-Distill-xl)              |
| LiT5-Distill-base-v2  | [castorini/LiT5-Distill-base-v2](https://huggingface.co/castorini/LiT5-Distill-base-v2)    |
| LiT5-Distill-large-v2 | [castorini/LiT5-Distill-large-v2](https://huggingface.co/castorini/LiT5-Distill-large-v2)  |
| LiT5-Distill-xl-v2    | [castorini/LiT5-Distill-xl-v2](https://huggingface.co/castorini/LiT5-Distill-xl-v2)        |
| LiT5-Score-base       | [castorini/LiT5-Score-base](https://huggingface.co/castorini/LiT5-Score-base)              |
| LiT5-Score-large      | [castorini/LiT5-Score-large](https://huggingface.co/castorini/LiT5-Score-large)            |
| LiT5-Score-xl         | [castorini/LiT5-Score-xl](https://huggingface.co/castorini/LiT5-Score-xl)                  |


## Expected Results

This table shows the expected results for reranking with BM25 first-stage retrieval

### DL19
| Model Name            | nDCG@10 |
|-----------------------|---------|
| LiT5-Distill-base     |    71.7 |
| LiT5-Distill-large    |    72.7 |
| LiT5-Distill-xl       |    72.3 |
| LiT5-Distill-base-v2  |    71.7 |
| LiT5-Distill-large-v2 |    73.3 |
| LiT5-Distill-xl-v2    |    73.0 |
| LiT5-Score-base       |    68.9 |
| LiT5-Score-large      |    72.0 |
| LiT5-Score-xl         |    70.0 |

### DL20
| Model Name            | nDCG@10 |
|-----------------------|---------|
| LiT5-Distill-base     |    68.0 |
| LiT5-Distill-large    |    70.0 |
| LiT5-Distill-xl       |    71.8 |
| LiT5-Distill-base-v2  |    66.7 |
| LiT5-Distill-large-v2 |    69.8 |
| LiT5-Distill-xl-v2    |    73.7 |
| LiT5-Score-base       |    66.2 |
| LiT5-Score-large      |    67.8 |
| LiT5-Score-xl         |    65.7 |

This table shows the expected results for reranking with SPLADE++ ED first-stage retrieval

### DL19
| Model Name            | nDCG@10 |
|-----------------------|---------|
| LiT5-Distill-base     |    74.6 |
| LiT5-Distill-large    |    76.8 |
| LiT5-Distill-xl       |    76.8 |
| LiT5-Distill-base-v2  |    78.3 |
| LiT5-Distill-large-v2 |    80.0 |
| LiT5-Distill-xl-v2    |    78.5 |
| LiT5-Score-base       |    68.4 |
| LiT5-Score-large      |    68.7 |
| LiT5-Score-xl         |    69.0 |

### DL20
| Model Name            | nDCG@10 |
|-----------------------|---------|
| LiT5-Distill-base     |    74.1 |
| LiT5-Distill-large    |    76.5 |
| LiT5-Distill-xl       |    76.7 |
| LiT5-Distill-base-v2  |    75.1 |
| LiT5-Distill-large-v2 |    76.6 |
| LiT5-Distill-xl-v2    |    80.4 |
| LiT5-Score-base       |    68.5 |
| LiT5-Score-large      |    73.1 |
| LiT5-Score-xl         |    71.0 |

## ✨ References

If you use LiT5, please cite the following paper: 
[[2312.16098] Scaling Down, LiTting Up: Efficient Zero-Shot Listwise Reranking with Seq2seq Encoder-Decoder Models](https://arxiv.org/abs/2312.16098)

```
@ARTICLE{tamber2023scaling,
  title   = {Scaling Down, LiTting Up: Efficient Zero-Shot Listwise Reranking with Seq2seq Encoder-Decoder Models},
  author  = {Manveer Singh Tamber and Ronak Pradeep and Jimmy Lin},
  year    = {2023},
  journal = {arXiv preprint arXiv: 2312.16098}
}
```

🙏 Acknowledgments

This repository borrows code from the original [FiD repository](https://github.com/facebookresearch/FiD), the [atlas repository](https://github.com/facebookresearch/atlas), and the [RankLLM repository](https://github.com/castorini/rank_llm)! 
