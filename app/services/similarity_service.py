import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embedding_path = 'data/embeddings/embeddings.npy'
ids_path = 'data/embeddings/images_ids.json'

### loadings embeddings

def load_embeddings():
    embeddings = np.load(embedding_path)
    
    with open(ids_path,'r') as f:
        image_ids = json.load(f)

    assert embeddings.shape[0] == len(image_ids), "embedding and id mismatch"
    return embeddings,image_ids

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings,axis=1,keepdims=True)
    return embeddings/norms

def find_similar(query_embedding:np.ndarray,embeddings:np.ndarray,image_ids:list,top_k:int=5):
    query_embedding = query_embedding.reshape(1,-1)

    similarities = cosine_similarity(query_embedding,embeddings)[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for idx in top_indices:
        results.append({
            'image_id':image_ids[idx],
            'similarity':float(similarities[idx])
        })

    return results