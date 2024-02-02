"""Functions to read .txt files, embed the contents and create a index"""

from typing import List, Optional
from datasets import Dataset, Features, Sequence, Value, load_from_disk
import os
from functools import partial
from dataclasses import dataclass, field
from pathlib import Path
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    HfArgumentParser,
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenizer,
)
import torch
import faiss
import numpy as np
from tqdm import tqdm
from src.config import RagExampleArguments, ProcessingArguments, IndexHnswArguments

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

def read_documents(txt_path) -> List[dict]:
    """Read .txt files in txt_path and returns a list of documents with title and text"""
    documents = []
    for filename in os.listdir(txt_path):
        if filename.endswith(".txt"):
            with open(os.path.join(txt_path, filename), "r") as f:
                documents.append({"title": filename, "text": f.read()})
    return documents

def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]

def split_documents(documents: List[dict], n=100, character=" ") -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    for document in documents:
        title = document['title']
        text = document['text']
        if text is not None:
            for passage in split_text(text, n=n, character=character):
                titles.append(title if title is not None else "")
                texts.append(passage)
    return {"title": titles, "text": texts}

def embed(documents: dict, ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast) -> dict:
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(
        documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt"
    )["input_ids"]
    embeddings = ctx_encoder(input_ids.to(device=device), return_dict=True).pooler_output
    return {"embeddings": embeddings.detach().cpu().numpy()}

def create_dataset(txt_path, output_passages_path="data/my_dataset", output_index_path="data/my_index.faiss", verbose=False) -> Dataset:
    """Create a hugging face dataset with embeddings and index of the text."""
    parser = HfArgumentParser((RagExampleArguments, ProcessingArguments, IndexHnswArguments))
    rag_example_args, processing_args, index_hnsw_args = parser.parse_args_into_dataclasses()
    if verbose:
        print("Creating dataset with faiss index.")
    documents = read_documents(txt_path)
    dataset = Dataset.from_dict(split_documents(documents))

    if verbose:
        print("Embedding the passages.")
    # And compute the embeddings
    ctx_encoder = DPRContextEncoder.from_pretrained(rag_example_args.dpr_ctx_encoder_model_name).to(device=device)
    ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(rag_example_args.dpr_ctx_encoder_model_name)
    new_features = Features(
        {"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))}
    )  # optional, save as float32 instead of float64 to save space
    dataset = dataset.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer),
        batched=True,
        batch_size=processing_args.batch_size,
        features=new_features,  
    )
    # And finally save your dataset
    #dataset.save_to_disk(output_passages_path)

    # Let's use the Faiss implementation of HNSW for fast approximate nearest neighbor search
    index = faiss.IndexHNSWFlat(index_hnsw_args.d, index_hnsw_args.m, faiss.METRIC_INNER_PRODUCT)
    dataset.add_faiss_index("embeddings", custom_index=index)

    # And save the index
    #dataset.get_index("embeddings").save(output_index_path)

    question_vec = embed(np.array([{"title": "", "text": "What is reiforcement learning?"}]), ctx_encoder, ctx_tokenizer)
    print(dataset.get_index("embeddings").search(question_vec))

    return dataset

def load_dataset(output_passages_path="data/my_dataset", output_index_path="data/my_index.faiss"):
    """Load already generated dataset with index."""
    dataset = load_from_disk(output_passages_path)  # to reload the dataset
    dataset.load_faiss_index("embeddings", output_index_path)  # to reload the index

    return dataset