[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# VISION TRANSFORMERS NEED REGISTERS
The vit model from the paper "VISION TRANSFORMERS NEED REGISTERS" that reaches SOTA for dense
visual prediction tasks, enables object discovery methods with larger model, and leads to smoother feature maps and attentions maps for downstream visual processing.

Register tokens enable interpretable attention maps in all vision transofrmers!



[Paper Link](https://arxiv.org/pdf/2309.16588.pdf)

# Appreciation
* Lucidrains
* Agorians

# Install
`pip install vit-rgts`

# Usage
```python
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

preds = v(img) # (1, 1000)
print(preds)
```

# Architecture
- Additional tokens to input sequence that cleanup low informative background areas of images

# License
MIT

# Citations

```
@misc{2309.16588,
Author = {Timothée Darcet and Maxime Oquab and Julien Mairal and Piotr Bojanowski},
Title = {Vision Transformers Need Registers},
Year = {2023},
Eprint = {arXiv:2309.16588},
}
```


# Todo
- [ ] Make a new training script
- [ ]