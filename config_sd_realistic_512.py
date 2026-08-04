from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
import os
from typing import Tuple

# Load environment variables from .env file
load_dotenv()

@dataclass(frozen=True)
class GenerationConfig:
    model_id: str = "stablediffusionapi/realistic-vision-v51"

    prompt: str = (
        "masterpiece, top quality, 1woman, 25-year-old brazilian woman, tan skin, subtle freckles, "
        "wavy dark hair, detailed natural smile, white linen shirt, green yellow wristband, "
        "sunny balcony background, tropical foliage, bokeh, morning light, shot on 85mm lens, f1.8, photorealistic, 8k uhd"
    )

    negative_prompt: str = (
        "deformed, bad anatomy, bad lighting, smooth skin, plastic, 3d render, illustration, cartoon, "
        "oversaturated, pale, bad eyes, poorly drawn face, disfigured, blur, bad composition"
    )

    dimensions: Tuple[int, int] = (512, 768)  # Aspect ratio 2:3 ideal para retratos no M1
    steps: int = 30
    guidance_scale: float = 6.0 # Guidance mais baixo evita superexposição e pele de cera
    seed: int = 101
    output_path: str = datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
    hf_token: str = os.getenv("HF_TOKEN")
