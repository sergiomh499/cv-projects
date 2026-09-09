---
title: "Masked Autoencoders (MAE): Asymmetric Vision Pretraining"
type: "Technique"
domain: "Vision Transformers, Self-Supervised Learning & Foundation Models"
tags:
  - technique
  - mae
  - masked-autoencoders
  - self-supervised-learning
  - vision-transformers
  - foundation-models
  - pretraining
status: evergreen
updated: 2026-09-09
aliases:
  - "Masked Autoencoders"
  - "MAE"
  - "Masked Image Modeling"
  - "MIM"
  - "Asymmetric Autoencoder"
---

# 🎭 Masked Autoencoders (MAE): Asymmetric Vision Pretraining

## 1. High-Level Concept & The Information Density Dilemma

In Natural Language Processing (NLP), Masked Language Modeling (e.g., BERT) pretrains transformers by randomly masking **$15\%$ of input words**:
- Words are human-generated semantic symbols with high information density.
- Masking just $15\%$ creates a challenging prediction task that requires deep grammatical, syntactic, and contextual reasoning.

### The Spatial Redundancy Trap in Computer Vision
When standard masking is applied to computer vision:
- Images are continuous physical signals with **extreme spatial redundancy**.
- Neighboring visual patches contain nearly identical color distributions, textures, and structures.
- If an image is masked with a small ratio (e.g., $15\%$), the network can easily interpolate missing patches using simple local linear continuity without learning any high-level semantic abstractions or 3D scene understanding!

---

### The Masked Autoencoder (MAE) Revolution (He et al., CVPR 2022)
**MAE** solves the visual pretraining bottleneck through two radical design principles:

#### 1. Extreme Masking Ratio ($75\text{--}80\%$)
By removing **three-quarters ($75\%$) of all visual patches at random**:
- Local pixel interpolation is completely disabled.
- To reconstruct the missing visual content, the network must develop a global, holistic understanding of objects, lighting geometry, perspective occlusion, and semantic category co-occurrences.

#### 2. Asymmetric Encoder-Decoder Architecture
- **Deep Heavyweight Encoder**: Operates **strictly on the small subset ($25\%$) of visible, unmasked patches**!
  - Mask tokens are *never* passed through the deep encoder.
  - Because self-attention complexity scales quadratically with sequence length ($\mathcal{O}(N^2)$), reducing sequence length by $4\times$ yields a **$16\times$ reduction ($0.25^2 = 0.0625$) in attention computation** and cut training VRAM by $>3\times$!
- **Lightweight Shallow Decoder**: Takes the encoded visible patch representations, inserts shared learnable `[mask]` tokens at all missing spatial positions, adds positional embeddings to the full sequence, and reconstructs raw pixel values using a fast 8-layer transformer.

```
Masked Autoencoder (MAE) Asymmetric Pipeline:

Raw Image (224x224) ---> Split into 14x14 = 196 Patches (16x16 pixels each)
                                |
                                v
               [ Random 75% Masking Operator ]
                                |
          +---------------------+---------------------+
          |                                           |
          v (25% Visible Patches, 49 tokens)          v (75% Masked Patches, 147 tokens)
[ Deep Heavyweight Encoder ]                   (Completely discarded during encoding!)
- Operates on 49 tokens only!
- 16x faster self-attention compute!
          |
          v
Encoded Latent Vectors (49 tokens)
          |
          v
[ Add Learnable [mask] Tokens to Restore Full Sequence: 196 tokens ]
          |
          v
[ Lightweight Shallow Decoder (e.g. 8 layers) ]
          |
          v
[ Reconstruct Normalized Raw Pixels for Masked Patches ]
          |
          v
Normalized MSE Loss (Supervises 75% masked patches only!)
```

---

## 2. Mathematical Formulation

### 2.1 The Patchify and Random Masking Operator
Let image $\mathbf{I} \in \mathbb{R}^{3 \times H \times W}$ be partitioned into non-overlapping patches of size $P \times P$ (e.g., $P = 16$).
The sequence length is $N = \frac{HW}{P^2}$. Each patch is flattened into vector $\mathbf{x}_i \in \mathbb{R}^{3P^2}$.

A random permutation $\pi \in \mathfrak{S}_N$ partitions patch indices $\{1, \dots, N\}$ into:
- Visible set $\mathcal{V} = \{ \pi(1), \dots, \pi(N_{\text{vis}}) \}$ where $N_{\text{vis}} = (1 - \rho) N$ ($\rho = 0.75$).
- Masked set $\mathcal{M} = \{ \pi(N_{\text{vis}} + 1), \dots, \pi(N) \}$ where $N_{\text{mask}} = \rho N$.

The encoder processes only the visible subset:

$$
\mathbf{Z}_{\text{vis}} = \text{Encoder}\left( \left[ \mathbf{x}_i \mathbf{W}_{\text{patch}} + \mathbf{E}_{\text{pos}}[i] \right]_{i \in \mathcal{V}} \right) \in \mathbb{R}^{N_{\text{vis}} \times d_{\text{enc}}}
$$

---

### 2.2 Decoder Sequence Reconstruction & Normalized Pixel Loss
The decoder sequence $\mathbf{Z}_{\text{dec}} \in \mathbb{R}^{N \times d_{\text{dec}}}$ is formed by projecting $\mathbf{Z}_{\text{vis}}$ to decoder dimension $d_{\text{dec}}$ and interleaving learnable mask tokens $\mathbf{e}_{\text{mask}} \in \mathbb{R}^{d_{\text{dec}}}$:

$$
\mathbf{Z}_{\text{dec}}[j] = \begin{cases} \mathbf{Z}_{\text{vis}}[i] \mathbf{W}_{\text{proj}} + \mathbf{E}_{\text{dec\_pos}}[j], & \text{if } j = \pi(i) \in \mathcal{V} \\ \mathbf{e}_{\text{mask}} + \mathbf{E}_{\text{dec\_pos}}[j], & \text{if } j \in \mathcal{M} \end{cases}
$$

The decoder predicts pixel values $\hat{\mathbf{p}}_j = \text{Decoder}(\mathbf{Z}_{\text{dec}})[j] \in \mathbb{R}^{3P^2}$.

#### Normalized Mean Squared Error Loss
For each masked patch $j \in \mathcal{M}$, ground-truth pixels $\mathbf{p}_j$ are normalized by their local patch mean $\mu_j$ and standard deviation $\sigma_j$:

$$
\tilde{\mathbf{p}}_j = \frac{\mathbf{p}_j - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}
$$

$$
\mathcal{L}_{\text{MAE}} = \frac{1}{|\mathcal{M}|} \sum_{j \in \mathcal{M}} \left\| \tilde{\mathbf{p}}_j - \hat{\mathbf{p}}_j \right\|_2^2
$$

Normalizing pixel targets prevents the loss from focusing on global illumination shifts, forcing the network to model spatial textures and high-frequency surface geometry.

---

## 3. PyTorch Reference Implementation

```python
import torch
import torch.nn as nn

class MaskedAutoencoderViT(nn.Module):
    """
    Masked Autoencoder (MAE) with asymmetric encoder-decoder design.
    Operates on 25% visible tokens only during encoding.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3,
                 encoder_dim: int = 768, decoder_dim: int = 512, mask_ratio: float = 0.75):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, encoder_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, encoder_dim))
        
        # Decoder
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))
        
        # Prediction head: predicts normalized raw pixels [3 * P * P]
        self.pred_head = nn.Linear(decoder_dim, patch_size * patch_size * in_channels, bias=True)

    def random_masking(self, x: torch.Tensor):
        """
        Splits sequence into 25% visible and 75% masked tokens.
        x: [B, N, d]
        """
        b, n, d = x.shape
        len_keep = int(n * (1.0 - self.mask_ratio))
        
        # Generate random noise for sorting
        noise = torch.rand(b, n, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep visible tokens only
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))
        
        # Binary mask: 1 = masked, 0 = visible
        mask = torch.ones([b, n], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        imgs: [B, 3, 224, 224]
        Returns: (loss, predicted_patches, mask)
        """
        # 1. Patchify and add positional embeddings
        x = self.patch_embed(imgs) # [B, d, 14, 14]
        x = x.flatten(2).transpose(1, 2) # [B, 196, d]
        x = x + self.pos_embed
        
        # 2. Masking: retain only 25% visible tokens
        x_vis, mask, ids_restore = self.random_masking(x) # [B, 49, d]
        
        # (Pass x_vis through Deep Encoder blocks here...)
        latent = x_vis 
        
        # 3. Project to decoder dimension
        dec_in = self.enc_to_dec(latent) # [B, 49, dec_dim]
        
        # 4. Insert learnable [mask] tokens to restore 196 length
        mask_tokens = self.mask_token.repeat(imgs.shape[0], ids_restore.shape[1] - dec_in.shape[1], 1)
        full_seq = torch.cat([dec_in, mask_tokens], dim=1) # [B, 196, dec_dim]
        full_seq = torch.gather(full_seq, dim=1, 
                                index=ids_restore.unsqueeze(-1).repeat(1, 1, dec_in.shape[-1]))
        full_seq = full_seq + self.dec_pos_embed
        
        # (Pass full_seq through Shallow Decoder blocks here...)
        pred_pixels = self.pred_head(full_seq) # [B, 196, 768]
        
        return pred_pixels, mask
```

---

## 4. Models in the Vault Utilizing MAE

- **[[architectures/vision-foundation-models/sam-2|SAM 2]] & [[architectures/vision-foundation-models/dinov2|DINOv2]]**: Backbone pretraining leveraging masked image modeling for dense feature extraction.
- **[[architectures/backbones-and-edge-efficiency/fastvit|FastViT]]**: Self-supervised distillation and structural pretraining.
- **[[architectures/vision-foundation-models/depth-anything-v2|Depth Anything v2]]**: Foundation monocular depth estimation leveraging representations initialized from masked pretraining.
