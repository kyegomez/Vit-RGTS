import torch 
from torch import nn
from einops import rearrange, repeat, unpack, pack
from einops.layers.torch import Rearrange
import torch.nn.functional as F

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def pos_emb_sincos_2d(
    h,
    w,
    dim,
    temperature: int = 10000,
    dtype = torch.float32
):
    y, x = torch.meshgrid(
        torch.arange(h), torch.arange(w), indexing="ij"
    )
    assert (dim % 4) == 0, "dimension must be divisible by 4"
    omega = torch.arange(dim // 4, dtype=dtype)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

class FeedForward(nn.Module):
    def __init__(
        self, 
        dim, 
        hidden_dim, 
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self, 
        dim, 
        heads = 8, 
        dim_head = 64, 
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim_head)
        self.norm_v = nn.LayerNorm(dim_head)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # #normalize key and values or known QK Normalization
        k = self.norm_k(k)
        v = self.norm_v(v)

        # should this be replaced?
        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # attn = self.attend(dots)
        # attn = self.dropout(attn)
        # out = torch.matmul(attn, v)
        
        # attn
        with torch.backends.cuda.sdp_kernel(enable_math=True):
            #attention
            out = F.scaled_dot_product_attention(q, k, v)

            #softmax
            out = self.attend(out)
            
            #dropout
            out = self.dropout(out)

            #rearrange to original shape
            out = rearrange(out, 'b h n d -> b n (h d)')

            #project out
            return self.to_out(out)
        
        
class Transformer(nn.Module):
    def __init__(
        self, 
        dim, 
        depth, 
        heads, 
        dim_head, 
        mlp_dim, 
        dropout = 0.
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            #layernorm before attention
            x = self.norm(x)
            
            #parallel
            x = x + attn(x) + ff(x)
        
        return self.norm(x)


class VitRGTS(nn.Module):
    """
    VitRGTS model from https://arxiv.org/abs/2106.14759

    Args:
    -------
    image_size: int
        Size of image
    patch_size: int
        Size of patch
    num_classes: int
        Number of classes
    dim: int
        Dimension of embedding
    depth: int
        Depth of transformer
    heads: int
        Number of heads
    mlp_dim: int
        Dimension of MLP
    pool: str
        Type of pooling
    channels: int
        Number of channels
    dim_head: int
        Dimension of head
    dropout: float
        Dropout rate
    emb_dropout: float
        Dropout rate for embedding
    
    Returns:
    --------
    torch.Tensor
        Predictions
    
    Methods:
    --------
    forward(img: torch.Tensor) -> torch.Tensor:
        Forward pass
    
    Architecture:
    -------------
    1. Input image is passed through a patch embedding layer
    2. Positional embedding is added
    3. Dropout is applied
    4. Transformer is applied
    5. Pooling is applied
    6. MLP head is applied
    7. Output is returned
    """
    def __init__(
        self, 
        *, 
        image_size, 
        patch_size, 
        num_classes, 
        dim, 
        depth, 
        heads, 
        mlp_dim,
        num_register_tokens: int = 4, 
        pool = 'cls', 
        channels = 3, 
        dim_head = 64, 
        dropout = 0., 
        emb_dropout = 0.
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                p1 = patch_height, 
                p2 = patch_width
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.register_tokens = nn.Parameter(
            torch.randn(num_register_tokens, dim)
        )
        self.pos_embedding = pos_emb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        batch, device = img.shape[0], img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device)

        r = repeat(self.register_tokens, 'n d -> b n d', b=batch)

        x, ps = pack([x, r], 'b * d ')

        x = self.transformer(x)

        x, _ = unpack(x, ps, 'b * d')

        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)