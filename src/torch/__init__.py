# Allow users to not install PyTorch if they only want to use non-torch architecures
try:
    import torch
except ImportError:
    Warning("PyTorch not installed. Some functionality may not work.")