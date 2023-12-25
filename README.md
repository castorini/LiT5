# LiT5 (List-in-T5) Reranking

## 📟 Instructions

We provide the scripts and data necessary to reproduce reranking results for [LiT5-Distill](LiT5-Distill.sh) and [LiT5-Score](LiT5-Score.sh) on DL19 and DL20 for BM25 and SPLADE++ ED first-stage retrieval. Note you may need to change the batchsize depending on your VRAM. We have observed that results may change slightly when the batchsize is changed. 

## Models

The following is a table of our models hosted on HuggingFace:

| Model Name            | Hugging Face Identifier/Link                                                         |
|-----------------------|--------------------------------------------------------------------------------------|
| LiT5-Distill-base     | [castorini/LiT5-Distill-base](https://huggingface.co/castorini/LiT5-Distill-base)    |
| LiT5-Distill-large    | [castorini/LiT5-Distill-large](https://huggingface.co/castorini/LiT5-Distill-large)  |
| LiT5-Distill-xl       | [castorini/LiT5-Distill-xl](https://huggingface.co/castorini/LiT5-Distill-xl)        |
| LiT5-Score-base       | [castorini/LiT5-Score-base](https://huggingface.co/castorini/LiT5-Score-base)        |
| LiT5-Score-large      | [castorini/LiT5-Score-large](https://huggingface.co/castorini/LiT5-Score-large)      |
| LiT5-Score-xl         | [castorini/LiT5-Score-xl](https://huggingface.co/castorini/LiT5-Score-xl)            |


## Expected Results

This table shows the expected results for reranking with BM25 first-stage retrieval

### DL19
| Model Name            | nDCG@10 |
|-----------------------|---------|
| LiT5-Distill-base     |    72.1 |
| LiT5-Distill-large    |    72.7 |
| LiT5-Distill-xl       |    72.2 |
| LiT5-Score-base       |    68.9 |
| LiT5-Score-large      |    72.0 |
| LiT5-Score-xl         |    70.0 |

### DL20
| Model Name            | nDCG@10 |
|-----------------------|---------|
| LiT5-Distill-base     |    68.5 |
| LiT5-Distill-large    |    70.3 |
| LiT5-Distill-xl       |    72.2 |
| LiT5-Score-base       |    66.2 |
| LiT5-Score-large      |    67.8 |
| LiT5-Score-xl         |    65.7 |

This table shows the expected results for reranking with SPLADE++ ED first-stage retrieval

### DL19
| Model Name            | nDCG@10 |
|-----------------------|---------|
| LiT5-Distill-base     |    74.8 |
| LiT5-Distill-large    |    77.2 |
| LiT5-Distill-xl       |    77.1 |

### DL20
| Model Name            | nDCG@10 |
|-----------------------|---------|
| LiT5-Distill-base     |    74.9 |
| LiT5-Distill-large    |    76.3 |
| LiT5-Distill-xl       |    76.8 |


🙏 Acknowledgments

This repository borrows code from the original [FiD repository](https://github.com/facebookresearch/FiD), the [atlas repository](https://github.com/facebookresearch/atlas), and the [RankLLM repository](https://github.com/castorini/rank_llm)! 
