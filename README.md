[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# VISION TRANSFORMERS NEED REGISTERS
The vit model from the paper "VISION TRANSFORMERS NEED REGISTERS" that reaches SOTA for dense
visual prediction tasks, enables object discovery methods with larger model, and leads to smoother feature maps and attentions maps for downstream visual processing.

Register tokens enable interpretable attention maps in all vision transformers!



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

# Dataset Strategy
Here is a table summarizing the key datasets mentioned in the paper along with their metadata and source links:

| Dataset | Type | Size | Tasks | Source |
|-|-|-|-|-|  
| ImageNet-1k | Image Classification | 1.2M images, 1000 classes | Pretraining | http://www.image-net.org/ |
| ImageNet-22k | Image Classification | 14M images, 21841 classes | Pretraining | https://github.com/google-research-datasets/ImageNet-21k-P |
| INaturalist (IN1k) | Image Classification | 437K images, 1000 classes | Evaluation | https://github.com/visipedia/inat_comp/tree/master/2018 |  
| Places205 (P205) | Image Classification | 2.4M images, 205 classes | Evaluation | http://places2.csail.mit.edu/index.html |
| Aircraft (Airc.) | Image Classification | 10K images, 100 classes | Evaluation | https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/ |
| CIFAR-10 (CF10) | Image Classification | 60K images, 10 classes | Evaluation | https://www.cs.toronto.edu/~kriz/cifar.html |  
| CIFAR-100 (CF100) | Image Classification | 60K images, 100 classes | Evaluation | https://www.cs.toronto.edu/~kriz/cifar.html |
| CUB-200-2011 (CUB) | Image Classification | 11.8K images, 200 classes | Evaluation | http://www.vision.caltech.edu/visipedia/CUB-200-2011.html |
| Caltech 101 (Cal101) | Image Classification | 9K images, 101 classes | Evaluation | http://www.vision.caltech.edu/Image_Datasets/Caltech101/ |
| Stanford Cars (Cars) | Image Classification | 16K images, 196 classes | Evaluation | https://ai.stanford.edu/~jkrause/cars/car_dataset.html | 
| Describable Textures (DTD) | Image Classification | 5640 images, 47 classes | Evaluation | https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html |
| MPI Sintel (Flow.) | Optical Flow | 1041 images | Evaluation | http://sintel.is.tue.mpg.de/ |
| Food-101 (Food) | Image Classification | 101K images, 101 classes | Evaluation | https://www.vision.ee.ethz.ch/datasets_extra/food-101/ |  
| Oxford-IIIT Pets (Pets) | Image Classification | 7349 images, 37 classes | Evaluation | https://www.robots.ox.ac.uk/~vgg/data/pets/ |
| SUN397 (SUN) | Scene Classification | 108K images, 397 classes | Evaluation | https://groups.csail.mit.edu/vision/SUN/ |
| PASCAL VOC 2007 (VOC) | Object Detection | 5011 images, 20 classes | Evaluation | http://host.robots.ox.ac.uk/pascal/VOC/voc2007/ |
| PASCAL VOC 2012 (VOC) | Object Detection | 11540 images, 20 classes | Evaluation | http://host.robots.ox.ac.uk/pascal/VOC/voc2012/ |
| COCO 2017 (COCO) | Object Detection | 118K images, 80 classes | Evaluation | https://cocodataset.org/#home |
| ADE20K (ADE20k) | Semantic Segmentation | 20K images, 150 classes | Evaluation | https://groups.csail.mit.edu/vision/datasets/ADE20K/ |
| NYU Depth V2 (NYUd) | Monocular Depth Estimation | 1449 images | Evaluation | https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html |

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
- [x] Make a new training script
- [x] Make a table of datasets used in the paper
- [ ] Make a blog article on architecture and applications
- [ ] Clean up operations, remove redundancy in attention, transformer, and vitgi
