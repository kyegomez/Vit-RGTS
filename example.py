import torch
from vit_rgts.main import VitRGTS

v = VitRGTS(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 256)
print(f'Input image shape: {img}') # Input image shape: torch.Size([1, 3, 256, 256])

preds = v(img) # (1, 1000)
print(f"Output tensors shape: {preds}") # Output tensors shape: torch.Size([1, 1000])
print(preds.shape)