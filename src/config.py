from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

@dataclass
class RagExampleArguments:
    csv_path: str = field(
        default=str(Path(__file__).parent / "test_data" / "my_knowledge_dataset.csv"),
        metadata={"help": "Path to a tab-separated csv file with columns 'title' and 'text'"},
    )
    question: Optional[str] = field(
        default=None,
        metadata={"help": "Question that is passed as input to RAG. Default is 'What does Moses' rod turn into ?'."},
    )
    rag_model_name: str = field(
        default="facebook/rag-sequence-nq",
        metadata={"help": "The RAG model to use. Either 'facebook/rag-sequence-nq' or 'facebook/rag-token-nq'"},
    )
    dpr_ctx_encoder_model_name: str = field(
        default="facebook/dpr-ctx_encoder-multiset-base",
        metadata={
            "help": (
                "The DPR context encoder model to use. Either 'facebook/dpr-ctx_encoder-single-nq-base' or"
                " 'facebook/dpr-ctx_encoder-multiset-base'"
            )
        },
    )


@dataclass
class ProcessingArguments:
    num_proc: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use to split the documents into passages. Default is single process."
        },
    )
    batch_size: int = field(
        default=16,
        metadata={
            "help": "The batch size to use when computing the passages embeddings using the DPR context encoder."
        },
    )


@dataclass
class IndexHnswArguments:
    d: int = field(
        default=768,
        metadata={"help": "The dimension of the embeddings to pass to the HNSW Faiss index."},
    )
    m: int = field(
        default=128,
        metadata={
            "help": (
                "The number of bi-directional links created for every new element during the HNSW index construction."
            )
        },
    )