from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
import os
from typing import Tuple

# Load environment variables from .env file
load_dotenv()

@dataclass(frozen=True)
class GenerationConfig:
    model_id: str = "RunDiffusion/Juggernaut-XL-v9"

    # SDXL entende linguagem natural corrida muito melhor que o SD 1.5
    prompt: str = (
        "close up portrait of a 25-year-old Brazilian woman, natural smile, soft skin texture, subtle freckles, "
        "light white linen top, sunny Rio de Janeiro apartment balcony background, blurry tropical foliage, "
        "soft morning daylight, shot on iPhone 15 Pro, 85mm lens, f1.8, shallow depth of field, 8k resolution"
    )

    negative_prompt: str = (
        "3d render, cgi, plastic skin, bad anatomy, deformed, smooth skin distortion, cartoon, doll, oversaturated, low quality"
    )

    # Resolução padrão SDXL (1024x1024)
    dimensions: Tuple[int, int] = (1024, 1024)

    # Modelos Lightning usam de 4 a 8 passos. Modelos SDXL padrão usam 20 a 30.
    steps: int = 30
    guidance_scale: float = 6.0
    seed: int = 42
    output_path: str = f"jug_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    hf_token: str = os.getenv("HF_TOKEN")
