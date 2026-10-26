from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from app.lstm_model import LSTMTextGenerator


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

# ==========================================
# Module 7 - LSTM Text Generation
# ==========================================

lstm_model = LSTMTextGenerator(corpus)

print("Training LSTM model...")
lstm_model.train(epochs=100)
print("LSTM model ready!")

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

# ==========================================
# Module 7 - LSTM Text Generation API
# ==========================================

@app.post("/generate_with_rnn")
def generate_with_rnn(request: TextGenerationRequest):
    try:
        generated_text = lstm_model.generate_text(
            start_word=request.start_word,
            length=request.length
        )

        return {"generated_text": generated_text}

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

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
from helper_lib.mnist_gan import MNISTGenerator


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
    
    

# ==========================================
# Assignment 3 - MNIST GAN Image Generation
# ==========================================

from fastapi.responses import Response


MNIST_GENERATOR_PATH = (
    Path(__file__).resolve().parent / "mnist_generator.pth"
)

mnist_generator_model = None


class MNISTGenerationRequest(BaseModel):
    num_images: int = 10


def load_mnist_generator():
    global mnist_generator_model

    if mnist_generator_model is None:

        if not MNIST_GENERATOR_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="MNIST Generator weights not found."
            )

        model = MNISTGenerator(noise_dim=100)

        model.load_state_dict(
            torch.load(
                MNIST_GENERATOR_PATH,
                map_location="cpu",
                weights_only=True
            )
        )

        model.eval()
        mnist_generator_model = model

    return mnist_generator_model


@app.post("/generate-mnist")
def generate_mnist(request: MNISTGenerationRequest):

    if request.num_images < 1 or request.num_images > 25:
        raise HTTPException(
            status_code=400,
            detail="num_images must be between 1 and 25."
        )

    model = load_mnist_generator()

    with torch.no_grad():

        noise = torch.randn(
            request.num_images,
            100
        )

        generated_images = model(noise)

        generated_images = (
            (generated_images + 1) / 2
        ).clamp(0, 1)

    # Create an image grid
    import math

    cols = min(5, request.num_images)
    rows = math.ceil(request.num_images / cols)

    grid_image = Image.new(
        "L",
        (cols * 28, rows * 28),
        color=0
    )

    for i in range(request.num_images):

        image_array = (
            generated_images[i, 0]
            .mul(255)
            .byte()
            .cpu()
            .numpy()
        )

        digit_image = Image.fromarray(
            image_array,
            mode="L"
        )

        x = (i % cols) * 28
        y = (i // cols) * 28

        grid_image.paste(digit_image, (x, y))

    # Enlarge for easier viewing
    grid_image = grid_image.resize(
        (cols * 112, rows * 112),
        resample=Image.Resampling.NEAREST
    )

    buffer = io.BytesIO()
    grid_image.save(buffer, format="PNG")

    return Response(
        content=buffer.getvalue(),
        media_type="image/png"
    )
