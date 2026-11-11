import io
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException, Response
from PIL import Image
from pydantic import BaseModel, Field
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from helper_lib.cifar_diffusion import CIFARDiffusion
from helper_lib.cifar_energy import CIFAREnergy

router = APIRouter(tags=["CIFAR-10 image generation"])
CHECKPOINTS = Path(__file__).resolve().parents[1] / "checkpoints"


class CIFARGenerationRequest(BaseModel):
    num_images: int = Field(default=4, ge=1, le=16)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)


@lru_cache(maxsize=2)
def load_generator(model_class, filename):
    try:
        state_dict = torch.load(
            CHECKPOINTS / filename, map_location="cpu", weights_only=True
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=f"Missing checkpoint {filename}.") from error
    model = model_class()
    model.load_state_dict(state_dict)
    return model.eval().requires_grad_(False)


def random_generator(seed):
    if seed is None:
        return None
    return torch.Generator().manual_seed(seed)


def png_response(images):
    pixels = images.add(1).mul(0.5).clamp(0, 1)
    grid = make_grid(pixels, nrow=min(len(images), 4), padding=2)
    image = to_pil_image(grid)
    image = image.resize((image.width * 3, image.height * 3), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Response(buffer.getvalue(), media_type="image/png")


PNG_RESPONSE = {200: {"content": {"image/png": {}}, "description": "CIFAR-10 image grid"}}


@router.post("/generate-diffusion", response_class=Response, responses=PNG_RESPONSE)
def generate_diffusion(request: CIFARGenerationRequest):
    model = load_generator(CIFARDiffusion, "cifar_diffusion.pth")
    images = model.sample(request.num_images, random_generator(request.seed))
    return png_response(images)


@router.post("/generate-energy", response_class=Response, responses=PNG_RESPONSE)
def generate_energy(request: CIFARGenerationRequest):
    model = load_generator(CIFAREnergy, "cifar_energy.pth")
    generator = random_generator(request.seed)
    initial_images = torch.rand((request.num_images, 3, 32, 32), generator=generator)
    images = model.sample(initial_images.mul(2).sub(1), steps=120, generator=generator)
    return png_response(images)
