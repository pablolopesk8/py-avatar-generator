import gc
import logging
import sys
import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from diffusers.utils import logging as diffusers_logging
from config_juggernaut_xl import GenerationConfig

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

def validate_prompt_limit(pipe: StableDiffusionXLPipeline, prompt: str) -> None:
    """
    Valida rigorosamente a contagem de tokens nos dois Tokenizers do SDXL.
    Se exceder o limite rígido de 77 tokens, emite alerta e encerra a execução.
    """
    tokens_1 = pipe.tokenizer(prompt)["input_ids"]
    tokens_2 = pipe.tokenizer_2(prompt)["input_ids"]
    
    len_1 = len(tokens_1)
    len_2 = len(tokens_2)

    print(f"[INFO] Contagem Tokenizer 1 (CLIP L/14): {len_1}/77 tokens")
    print(f"[INFO] Contagem Tokenizer 2 (OpenCLIP bigG/14): {len_2}/77 tokens")

    if len_1 > 77 or len_2 > 77:
        print("\n" + "=" * 60)
        print("[ALERTA CRÍTICO] Limite de tokens excedido!")
        print(f"-> Tokenizer 1: {len_1} tokens (Limite: 77)")
        print(f"-> Tokenizer 2: {len_2} tokens (Limite: 77)")
        print("Ajuste o prompt para remover caracteres/palavras excedentes.")
        print("=" * 60 + "\n")
        raise ValueError("LIMITE DE TOKENS EXCEDIDO")

def build_pipeline(config: GenerationConfig, device: torch.device) -> StableDiffusionXLPipeline:
    """Carrega o pipeline com otimizações extremas de memória para 8GB M1."""
    print(f"[INFO] Carregando modelo SDXL '{config.model_id}' em float16...")
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config.model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        token=config.hf_token,
    )

    pipe.vae.to(dtype=torch.float16)

    # Scheduler otimizado para SDXL
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, 
        timestep_spacing="leading"
    )

    # --- OTIMIZAÇÕES CRÍTICAS PARA 8GB RAM M1 ---
    # Transfere submódulos (TextEncoder1, TextEncoder2, UNet, VAE) para a GPU/MPS somente no momento do cálculo
    pipe.enable_model_cpu_offload()

    # Slice de Atenção: reduz pico de memória no cálculo do Attention Map
    pipe.enable_attention_slicing(slice_size="auto")

    # Decodificação VAE em chunks para evitar estourar RAM no pós-processamento
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    return pipe

def generate_character(config: GenerationConfig) -> None:
    device = get_mps_device()
    pipe = None

    try:
        pipe = build_pipeline(config, device)

        validate_prompt_limit(pipe, config.prompt)
        
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
