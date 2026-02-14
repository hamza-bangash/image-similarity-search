from fastapi import FastAPI,UploadFile,File,HTTPException
from PIL import Image
import numpy as np

from app.services.blob_service import download_embeddings
from app.services.embedding_service import load_model, extract_embedding
from app.services.similarity_service import (normalize_embeddings,find_similar)

from app.utils.image_utils import preprocess_image

app = FastAPI(title="Image Similarity Search Api")


@app.on_event("startup")
def startup():

    app.state.model = load_model()

    embeddings, image_ids = download_embeddings()
    app.state.embeddings = normalize_embeddings(embeddings)
    app.state.image_ids = image_ids
    app.state.num_embeddings = app.state.embeddings.shape[0]



@app.post("/search")
async def search_image(
    file: UploadFile = File(...),
    top_k: int = 5
):
    if top_k <= 0:
        raise HTTPException(status_code=400, detail="number must be greater than zero")
    top_k = min(top_k, app.state.num_embeddings)

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")



    image_tensor = preprocess_image(image)
    query_embedding = extract_embedding(app.state.model, image_tensor)

    # Normalize query embedding
    norm = np.linalg.norm(query_embedding)
    if norm == 0:
        raise HTTPException(status_code=400, detail="embedding norm is zero")

    query_embedding = query_embedding / norm


    results = find_similar(
        query_embedding=query_embedding,
        embeddings=app.state.embeddings,
        image_ids=app.state.image_ids,
        top_k=top_k
    )

    return {
        "top_k": top_k,
        "results": results
    }

@app.get("/health")
def health():
    return {"status": "ok"}
