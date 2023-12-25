import torch
import transformers
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler

from src.options import Options
import src.data
import src.model

import copy
import json 
from typing import List, Union, Dict, Any, Tuple

def evaluate(model, dataset, dataloader, tokenizer, opt):
    generated_permutations = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, passage_ids, passage_mask, query) = batch
            passage_ids = passage_ids.contiguous().view(passage_ids.size(0), -1)
            passage_mask = passage_mask.contiguous().view(passage_mask.size(0), -1)

            outputs = model.generate(
                input_ids=passage_ids.cuda(),
                attention_mask=passage_mask.cuda(),
                max_length=opt.answer_maxlength,
                do_sample=False
            )
                                        
            for k, o in enumerate(outputs):
                output = tokenizer.decode(o, skip_special_tokens=True)
                generated_permutations.append(output)
    return generated_permutations

def clean_response(response: str) -> str:
                new_response = ""
                for c in response:
                    if not c.isdigit():
                        new_response += " "
                    else:
                        new_response += c
                new_response = new_response.strip()
                return new_response
def remove_duplicate(response: List[int]) -> List[int]:
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response


if __name__ == "__main__":
    torch.manual_seed(0)
    transformers.set_seed(0)
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()

    tokenizer = transformers.T5Tokenizer.from_pretrained(opt.model_path, return_dict=False, legacy=False, use_fast=True)

    collator_function = src.data.Collator(opt.text_maxlength, tokenizer, batch_size=opt.batch_size, n_passages=opt.n_passages, suffix= " Relevance Ranking: ")
    eval_examples = src.data.load_data(opt.eval_data)

    model_class = src.model.FiD
    model = model_class.from_pretrained(opt.model_path, from_flax=False).cuda().eval()
    if opt.bfloat16:
        model = model.bfloat16()
    
    for query in eval_examples:
        query['ctxs'] = query['ctxs'][:opt.n_rerank_passages]
    
    stride = opt.stride
    window_size = opt.n_passages

    print("Start Inference")
    for passes in range(1):
        for window_start_idx in range(opt.n_rerank_passages - window_size, -1, -stride):
            eval_dataset = src.data.Dataset(
                eval_examples, 
                opt.n_passages,
                start_pos=window_start_idx,
                question_prefix='Search Query:',
                passage_prefix='Passage:',
                passage_numbering=True
            )
            print('Reranking passages:', window_start_idx, 'to', window_start_idx+window_size)

            eval_sampler = SequentialSampler(eval_dataset) 
            eval_dataloader = DataLoader(
                eval_dataset, 
                sampler=eval_sampler, 
                batch_size=opt.batch_size,
                num_workers=4, 
                collate_fn=collator_function
            )
            
            generated_permutations = evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)
            eval_examples = eval_dataset.data
            
            for i in range(len(eval_examples)):
                query_dict = eval_examples[i]
                permutation = generated_permutations[i]
                
                resort_passages = copy.deepcopy(query_dict['ctxs'][window_start_idx:window_start_idx+window_size])
                if len(resort_passages) > 0:
                    response = clean_response(permutation)
                    response = [int(x) - 1 for x in response.split()]
                    response = remove_duplicate(response)
                    original_rank = [tt for tt in range(len(resort_passages))]
                    response = [ss for ss in response if ss in original_rank]
                    response = response + [tt for tt in original_rank if tt not in response]
                    for j, x in enumerate(response):
                        query_dict['ctxs'][j + window_start_idx] = resort_passages[x]

    with open(opt.runfile_path, 'w') as f:
        for query in eval_dataset.data:
            rank = 1
            for passage in query['ctxs']:
                if 'docid' in passage.keys(): 
                    f.write(" ".join([query['id'], "Q0", str(passage['docid']), str(rank), str(1/rank), "RankFiD\n"]))
                    rank+=1


