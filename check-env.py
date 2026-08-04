import sys
import gc
import torch

def check_mps_environment() -> None:
    print(f"=== Diagnóstico do Ambiente Python ===")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}\n")

    # 1. Verifica disponibilidade do MPS (Apple Silicon GPU)
    if not torch.backends.mps.is_available():
        print("[ERRO] MPS não está disponível neste ambiente PyTorch.")
        print("Certifique-se de estar usando uma versão do PyTorch compilada para arm64.")
        sys.exit(1)
    
    if not torch.backends.mps.is_built():
        print("[ERRO] O PyTorch foi instalado sem suporte aos binários do MPS.")
        sys.exit(1)

    device = torch.device("mps")
    print(f"[OK] backend MPS detectado e operacional: {device}")

    # 2. Teste funcional de alocação de Tensor no MPS
    try:
        # Teste usando float16 para economizar VRAM no M1 8GB
        x = torch.ones((1000, 1000), dtype=torch.float16, device=device)
        y = x * 2.0
        
        # Garante a sincronização dos comandos executados na GPU Metal
        torch.mps.synchronize()
        print("[OK] Alocação e computação de Tensors no MPS (float16) concluídas com sucesso.")
    except Exception as e:
        print(f"[ERRO] Falha ao processar tensores na GPU Metal: {e}")
        sys.exit(1)

    # 3. Otimizações de memória para o limite de 8GB RAM unificada
    print("\n=== Otimizações de Memória (M1 8GB) ===")
    
    # Executa a coleta de lixo do Python e o descarte de cache do MPS
    gc.collect()
    torch.mps.empty_cache()
    
    print("[OK] `torch.mps.empty_cache()` e `gc.collect()` executados.")
    print("\nAmbiente pronto para pipelines de IA generativa com PyTorch e Diffusers!")

if __name__ == "__main__":
    check_mps_environment()
