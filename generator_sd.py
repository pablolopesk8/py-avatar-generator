import gc
import logging
import os
import ssl
import sys
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import logging as diffusers_logging
from config_sd_realistic_512 import GenerationConfig

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Silencia avisos informativos/warnings de licença do Diffusers
diffusers_logging.set_verbosity_error()
logging.getLogger("diffusers").setLevel(logging.ERROR)

def get_mps_device() -> torch.device:
    """Valida e retorna o dispositivo MPS (Metal Performance Shaders)."""
    if not torch.backends.mps.is_available():
        print("[ERRO] MPS não está disponível. Certifique-se de estar usando PyTorch arm64.")
        sys.exit(1)
    return torch.device("mps")

def clear_vram() -> None:
    """Executa limpeza explícita da RAM/VRAM unificada no macOS."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def build_pipeline(config: GenerationConfig, device: torch.device) -> StableDiffusionPipeline:
    """Carrega o pipeline com otimizações extremas de memória para 8GB M1."""
    print(f"[INFO] Carregando modelo '{config.model_id}' em float32...")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        config.model_id,
        torch_dtype=torch.float32,
        use_safetensors=True,
        token=config.hf_token,
        safety_checker=None,
        feature_extractor=None
    )

    pipe.vae.to(dtype=torch.float32)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, 
        use_karras_sigmas=True
    )

    # 1. Ativa offload de submodelos para a CPU (Text Encoder, UNet, VAE) conforme necessário
    pipe.enable_sequential_cpu_offload(gpu_id=0)

    # 2. Slice de Atenção: reduz pico de memória no cálculo do Attention Map
    pipe.enable_attention_slicing(slice_size="auto")

    # 3. Decodificação VAE em chunks para evitar estourar RAM no pós-processamento
    pipe.vae.enable_slicing()

    return pipe

def generate_character(config: GenerationConfig) -> None:
    device = get_mps_device()
    pipe = None

    try:
        pipe = build_pipeline(config, device)
        generator = torch.Generator(device="cpu").manual_seed(config.seed)

        print("[INFO] Gerando imagem do personagem...")
        with torch.no_grad():
            output = pipe(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                width=config.dimensions[0],
                height=config.dimensions[1],
                num_inference_steps=config.steps,
                guidance_scale=config.guidance_scale,
                generator=generator
            )

        image = output.images[0]
        image.save(config.output_path)
        print(f"[SUCESSO] Personagem renderizado salvo em: {config.output_path}")

    except Exception as e:
        print(f"[ERRO] Falha durante a geração: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup agressivo para reter estabilidade do macOS
        if pipe is not None:
            del pipe
        clear_vram()
        print("[INFO] Memória liberada.")

if __name__ == "__main__":
    cfg = GenerationConfig()
    generate_character(cfg)
