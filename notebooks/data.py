import numpy as np

import os
from llama_index.embeddings import OpenAIEmbedding, HuggingFaceEmbedding


EMBEDDING_DIMENSIONS = {
    'sentence-transformers/all-mpnet-base-v2': 768,
}

def get_embedding_model(model_name, embed_batch_size=100):
    if model_name == "text-embedding-ada-002":
            return OpenAIEmbedding(
                model=model_name,
                embed_batch_size=embed_batch_size,
                api_key=os.environ["OPENAI_API_KEY"])
    else:
        return HuggingFaceEmbedding(
            model_name=model_name,
            embed_batch_size=embed_batch_size
        )
    