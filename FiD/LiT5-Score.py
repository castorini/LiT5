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
    if opt.write_crossattention_scores:
        model.overwrite_forward_crossattention()
        model.reset_score_storage() 

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, passage_ids, passage_mask, query) = batch
            passage_ids = passage_ids.contiguous().view(passage_ids.size(0), -1)
            passage_mask = passage_mask.contiguous().view(passage_mask.size(0), -1)

            if opt.write_crossattention_scores:
                model.reset_score_storage()

            outputs = model.generate(
                input_ids=passage_ids.cuda(),
                attention_mask=passage_mask.cuda(),
                max_length=opt.answer_maxlength,
                do_sample=False
            )
            
            # need to zero out scores after EOS token. This is needed when batching results in sequences with different lengths.
            output_sequence_lengths = [] 
            for output in outputs:
                length = 0
                for token in output:
                    if token == 1: # EOS token
                        break
                    length += 1
                output_sequence_lengths.append(length)

            if opt.write_crossattention_scores:
                query_mask_reader = (
                    tokenizer.batch_encode_plus(
                        query,
                        max_length=opt.text_maxlength,
                        padding="longest",
                        truncation=True,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["attention_mask"]
                    .bool()
                    .cuda()
                )

                crossattention_scores = model.get_crossattention_scores(opt.n_passages,
                    mask=passage_mask.cuda(),
                    ids=passage_ids.cuda(),
                    mask_query=query_mask_reader.cuda(),
                    output_sequence_lengths=output_sequence_lengths)
                                        
            for k, o in enumerate(outputs):
                example = dataset.data[idx[k]]
                if opt.write_crossattention_scores:
                    for j in range(min(len(example['ctxs']), opt.n_passages)):
                        for key in crossattention_scores:
                            example['ctxs'][j][key] = crossattention_scores[key][k, j].item()



if __name__ == "__main__":
    torch.manual_seed(0)
    transformers.set_seed(0)
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()

    tokenizer = transformers.T5Tokenizer.from_pretrained(opt.model_path, return_dict=False, legacy=False, use_fast=True)

    collator_function = src.data.Collator(opt.text_maxlength, tokenizer, batch_size=opt.batch_size, n_passages=opt.n_passages)
    
    eval_examples = src.data.load_data(
        opt.eval_data, 
    )
    eval_dataset = src.data.Dataset(
        eval_examples, 
        opt.n_passages,
        start_pos=0,
    )

    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=opt.batch_size,
        num_workers=4, 
        collate_fn=collator_function
    )
    
    model_class = src.model.FiD
    model = model_class.from_pretrained(opt.model_path, from_flax=False).cuda().eval()
    if opt.bfloat16:
        model = model.bfloat16()

    print("Start Inference")

    evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)

    with open(opt.runfile_path, 'w') as f:
        for query in eval_dataset.data:
            sort_passages = []
            for passage in query['ctxs']:
                if 'docid' in passage.keys() and opt.sort_key in passage.keys():
                    sort_passages.append(passage)
            sort_passages = sorted(sort_passages, key=lambda x: x[opt.sort_key], reverse=True)
            
            rank = 1
            for passage in sort_passages: 
                f.write(" ".join([query['id'], "Q0", str(passage['docid']), str(rank), str(1/rank), "ScoreFiD\n"]))
                rank+=1

