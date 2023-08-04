import asyncio
from io import BytesIO
from time import perf_counter_ns
from typing import List

import aiohttp
import numpy as np
from fashion_clip.fashion_clip import FashionCLIP
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

app = FastAPI()


# Request body model for /createEmbedding
class CreateEmbeddingRequest(BaseModel):
    image_urls: List[str]


# Helper function to download an image asynchronously
async def download_image(session: aiohttp.ClientSession, url) -> Image:
    """
    Downloads an image from the given URL using the provided aiohttp session.

    Args:
        session (aiohttp.ClientSession): The aiohttp session to use for the download.
        url (str): The URL of the image to download.

    Returns:
        Image: The downloaded image.
    """
    async with session.get(url) as response:
        image = Image.open(BytesIO(await response.read()))
        return image


@app.get("/")
def status():
    """
    A function that returns the status of the API.

    :return: A dictionary containing the status of the API.
    :rtype: dict
    """
    return {"status": "OK"}


@app.post("/createEmbedding")
async def main(body: CreateEmbeddingRequest):
    """
    Create an embedding for an image.

    Parameters:
        image_path (str, optional): The path to the image file. If not provided, a default image path is used.

    Returns:
        JSONResponse: A JSON response containing the image embedding.
    """

    begin = perf_counter_ns()

    # Load fashion-clip Model
    model = FashionCLIP("fashion-clip")

    # Download the images asyncronously
    images = []
    async with aiohttp.ClientSession() as session:
        tasks = [download_image(session, url) for url in set(body.image_urls)]
        images = await asyncio.gather(*tasks)

    # Encode the image to generate the embedding
    embeddings = model.encode_images(images, batch_size=1)

    # Add all the numpy embeddings together and take the average
    embedding = np.divide(np.sum(embeddings, axis=0), len(embeddings)).tolist()

    end = perf_counter_ns()

    # Return the embedding as a JSON response
    response_data = {
        "embedding": embedding,
        "images": len(embeddings),
        "dim": len(embedding),
        "time": f"{(end - begin) / 1_000_000:.2f} ms",
    }
    return JSONResponse(response_data)
