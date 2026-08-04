from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
import os
from typing import Tuple

# Load environment variables from .env file
load_dotenv()

@dataclass(frozen=True)
class GenerationConfig:
    # Juggernaut-XL v9 / Lightning ou Refiner otimizado
    # Usaremos a versão Lightning/FP16 que exige menos passos e reduz uso de VRAM
    model_repo: str = "RunDiffusion/Juggernaut-XL-v9"

    # SDXL entende linguagem natural corrida muito melhor que o SD 1.5
    prompt: str = (
        "close up portrait of a 25-year-old Brazilian woman, natural smile, soft skin texture, subtle freckles, "
        "wearing a light white linen top, sunny Rio de Janeiro apartment balcony background, blurry tropical foliage, "
        "soft morning daylight, shot on iPhone 15 Pro, 85mm lens, f/1.8, shallow depth of field, photorealistic, 8k resolution"
    )

    negative_prompt: str = (
        "3d render, cgi, plastic skin, bad anatomy, deformed, smooth skin distortion, cartoon, doll, oversaturated, low quality"
    )

    # Modelos Lightning usam de 4 a 8 passos. Modelos SDXL padrão usam 20 a 30.
    steps: int = 25
    guidance_scale: float = 1.5  # SDXL Lightning usa CFG baixo (1.5-2.0)
    seed: int = 42
    output_path: str = f"sdxl_realistic_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    hf_token: str = os.getenv("HF_TOKEN")
