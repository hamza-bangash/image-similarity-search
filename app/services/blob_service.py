import os
import json
import numpy as np
from azure.storage.blob import BlobServiceClient
from io import BytesIO

##### downloading embeddings from azure blobs and images ids
def download_embeddings():
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = 'image'

    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(container_name)


    blob_client = container_client.get_blob_client("embeddings.npy")
    stream = BytesIO()
    stream.write(blob_client.download_blob().readall())
    stream.seek(0)
    embeddings = np.load(stream)

    # Download image_ids.json
    blob_client = container_client.get_blob_client("image_ids.json")
    ids_bytes = blob_client.download_blob().readall()
    image_ids = json.loads(ids_bytes.decode("utf-8"))


    return embeddings , image_ids

