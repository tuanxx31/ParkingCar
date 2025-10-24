import torch
print("PyTorch version:", torch.__version__)
print("PyTorch built with CUDA:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
