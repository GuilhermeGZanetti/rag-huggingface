from typing import List, Optional
from datasets import Dataset, Features, Sequence, Value, load_from_disk
import os
from functools import partial
from dataclasses import dataclass, field
from pathlib import Path
import torch
import faiss
from tqdm import tqdm

from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    HfArgumentParser,
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenizer,
)


from src.config import RagExampleArguments, ProcessingArguments, IndexHnswArguments

def load_rag(dataset, verbose=False):  
    parser = HfArgumentParser((RagExampleArguments, ProcessingArguments, IndexHnswArguments))
    rag_example_args, _, _ = parser.parse_args_into_dataclasses()

    if verbose:
        print("Loading RAG model ", rag_example_args.rag_model_name)

    retriever = RagRetriever.from_pretrained(
        rag_example_args.rag_model_name, index_name="custom", indexed_dataset=dataset
    )
    model = RagSequenceForGeneration.from_pretrained(rag_example_args.rag_model_name, retriever=retriever)
    tokenizer = RagTokenizer.from_pretrained(rag_example_args.rag_model_name)

    return (model, tokenizer)

def generate_answer(rag_sys, question):
    model, tokenizer = rag_sys

    #input_ids = tokenizer.question_encoder(question, return_tensors="pt")["input_ids"]
    #generated = model.generate(input_ids, max_new_tokens=500)
    #print("Generated: ", generated)
    #generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    input_dict = tokenizer.prepare_seq2seq_batch(question, return_tensors="pt") 

    generated = model.generate(input_ids=input_dict["input_ids"], max_new_tokens=500) 
    generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    return generated_string