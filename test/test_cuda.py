import torch

def test_cuda():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("-" * 30)
        # CUDA version PyTorch was compiled with
        print(f"PyTorch built with CUDA version: {torch.version.cuda}")
        
        # cuDNN version
        if torch.backends.cudnn.is_available():
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
        else:
            print("cuDNN is not available.")
        print("-" * 30)
        
        print("CUDA device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
        print("CUDA capability:", torch.cuda.get_device_capability())

        # Run a small tensor test on GPU
        x = torch.rand(3, 3).to("cuda")
        y = torch.rand(3, 3).to("cuda")
        z = torch.matmul(x, y)
        print("\nMatrix multiplication successful on GPU. Result:")
        print(z)
    else:
        print("CUDA is NOT available. Using CPU only.")

if __name__ == "__main__":
    test_cuda()
 