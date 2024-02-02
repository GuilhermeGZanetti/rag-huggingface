# rag-huggingface
A end to end rag library based on Hugging Face


python utils/consolidate_rag_checkpoint.py \
    --model_type rag_sequence \
    --generator_name_or_path mistralai/Mistral-7B-v0.1 \
    --question_encoder_name_or_path facebook/dpr-question_encoder-single-nq-base \
    --dest models/mistral7