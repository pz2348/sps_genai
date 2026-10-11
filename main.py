from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel


app = FastAPI()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)

embedding_model = EmbeddingModel()


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class EmbeddingRequest(BaseModel):
    word: str


class SimilarityRequest(BaseModel):
    word1: str
    word2: str

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length
    )
    return {"generated_text": generated_text}

@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = embedding_model.calculate_embedding(request.word)
    return {
        "word": request.word,
        "embedding": embedding
    }


@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    similarity = embedding_model.calculate_similarity(
        request.word1,
        request.word2
    )
    return {
        "word1": request.word1,
        "word2": request.word2,
        "similarity": similarity
    }
    
# ==========================================
# Assignment 2 - CNN Image Classification
# ==========================================

import io
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from fastapi import UploadFile, File, HTTPException

from helper_lib.model import get_model


# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# Load the trained CNN model
MODEL_PATH = Path(__file__).resolve().parent / "cifar10_cnn.pth"

cnn_model = None

image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


def load_cnn_model():
    global cnn_model

    if cnn_model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="CNN model weights not found."
            )

        model = get_model("CIFAR10CNN")

        model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        )

        model.eval()
        cnn_model = model

    return cnn_model


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, PNG, or WebP image."
        )

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = image_transform(image).unsqueeze(0)

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image."
        )

    model = load_cnn_model()

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    return {
        "predicted_class": CIFAR10_CLASSES[predicted.item()],
        "confidence": round(confidence.item(), 4)
    }