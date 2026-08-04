from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

@dataclass(frozen=True)
class GenerationConfig:
    # SD 1.5 é ideal para 8GB de RAM. Se usar SDXL, mantenha a resolução em 1024x1024 no máximo.
    model_id: str = "runwayml/stable-diffusion-v1-5"
    # model_id: str = "RunDiffusion/Juggernaut-XL-v9"
    model_id: str = "stablediffusionapi/realistic-vision-v51"

    # prompt: str = (
    #     "raw photo, close-up portrait of a 25-year-old woman, natural smile, soft skin texture, subtle freckles, "
    #     "white linen top, sunny balcony background, soft daylight, 85mm lens, f/1.8, shallow depth of field, high detail"
    # )
    prompt: str = (
        "masterpiece, top quality, 1woman, 25-year-old brazilian woman, tan skin, subtle freckles, "
        "wavy dark hair, detailed natural smile, white linen shirt, green yellow wristband, "
        "sunny balcony background, tropical foliage, bokeh, morning light, shot on 85mm lens, f1.8, photorealistic, 8k uhd"
    )

    # negative_prompt: str = (
    #     "deformed, distorted, disfigured, bad anatomy, bad eyes, plastic skin, 3d render, cgi, cartoon, doll, oversaturated, bad face"
    # )
    negative_prompt: str = (
        "deformed, bad anatomy, bad lighting, smooth skin, plastic, 3d render, illustration, cartoon, "
        "oversaturated, pale, bad eyes, poorly drawn face, disfigured, blur, bad composition"
    )

    dimensions: Tuple[int, int] = (512, 768)  # Aspect ratio 2:3 ideal para retratos no M1
    steps: int = 30
    guidance_scale: float = 6.0 # Guidance mais baixo evita superexposição e pele de cera
    seed: int = 101
    output_path: str = datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
    hf_token: str = "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"
