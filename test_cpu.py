import torch

def test_gpu_availability():
    print("=== PyTorch GPU (CUDA) Diagnostic ===")
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Architecture: {torch.cuda.get_device_capability(0)}")
        
        # Test basic tensor allocation
        try:
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            print("Successfully allocated tensor on GPU:", x)
            print("Device of tensor:", x.device)
        except Exception as e:
            print(f"Error allocating tensor on GPU: {e}")
    else:
        print("CRITICAL: PyTorch cannot detect a GPU (CUDA is False).")
        print("This will cause inference to run slowly on the CPU.")

if __name__ == "__main__":
    test_gpu_availability()
