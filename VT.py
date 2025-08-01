import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, inChannels, patchSize, embeddim, nopatches , *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inChannel = inChannels
        self.patchSize = patchSize
        self.embeddim = embeddim
        self.convolve    = nn.Conv2d(in_channels = inChannels , out_channels = embeddim, kernel_size = patchSize, stride = patchSize)
        self.flatten     = nn.Flatten(start_dim = 1 , end_dim = 2)
        self.tokenEmbeddings = nn.Parameter(torch.rand((1 , 1 , embeddim), requires_grad = True))
        self.positionEmbeddings = nn.Parameter(torch.rand((1 , nopatches + 1 , embeddim), requires_grad = True))
    
    def forward(self, x):
        x = self.convolve(x)
        x = x.permute((0 , 2 , 3 , 1))
        x = self.flatten(x)
        b = x.shape[0]
        x = torch.cat((self.tokenEmbeddings.expand(b, -1 ,-1) , x), dim = 1)
        x = x + self.positionEmbeddings
        return x
    
class MSABlock(nn.Module):
    def __init__(self, embeddim, nHeads = 12 , dropout = 0.0 , *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm   = nn.LayerNorm(normalized_shape = embeddim)
        self.attention = nn.MultiheadAttention(num_heads = nHeads , dropout = dropout , embed_dim = embeddim , batch_first = True)

    def forward(self, x):
        x = self.norm(x)
        x , _ = self.attention(query = x, key = x , value = x, need_weights = False)
        return x
    
class MLPBlock(nn.Module):
    def __init__(self, indim , interdim, outdim , dropout = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interdim = interdim
        self.indim = indim
        self.outdim = outdim
        self.dropout = dropout
        self.norm = nn.LayerNorm(normalized_shape = self.indim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features = self.indim , out_features = self.interdim),
            nn.GELU(),
            nn.Dropout(p = self.dropout),
            nn.Linear(in_features = self.interdim, out_features = self.outdim),
            nn.Dropout(p = self.dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.mlp(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, embeddim , mlpdim , mlpDropout = 0.1, msaDropout = 0 ,noHeads = 12 , *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddim = embeddim
        self.mlpdim = mlpdim
        self.mlpDropout = mlpDropout
        self.msaDropout = msaDropout
        self.noHeads = noHeads
        self.mlp = MLPBlock(indim = self.embeddim , outdim = self.embeddim , interdim = self.mlpdim, dropout = self.mlpDropout)
        self.msa = MSABlock(embeddim = self.embeddim, nHeads = self.noHeads, dropout = self.msaDropout)

    def forward(self, x):
        x = self.msa(x) + x
        x = self.mlp(x) + x
        return x
    
class VIT(nn.Module):
    def __init__(self, 
                 imageHeight , 
                 imageWidth , 
                 nClasses ,
                 nBlocks    = 12,
                 patchSize  = 16, 
                 channels   = 3, 
                 mlpdropout = 0.1 , 
                 msadropout = 0.0 ,
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.imageHeight = imageHeight
        self.imageWidth  = imageWidth
        self.nClasses    = nClasses
        self.nBlocks     = nBlocks
        self.patchSize   = patchSize
        self.channels    = channels
        self.embeddim    = int(self.channels * (self.patchSize ** 2))
        self.noPatches   = int((self.imageHeight * self.imageWidth) / (self.patchSize ** 2))
        self.mlpdropout  = mlpdropout
        self.msadropout  = msadropout
        self.mlpsize     = (self.patchSize ** 2) * 12
        self.patches     = PatchEmbedding(
            inChannels   = self.channels , 
            patchSize    = patchSize, 
            nopatches    = self.noPatches , 
            embeddim     = self.embeddim)
        self.transformer = nn.Sequential(*[TransformerBlock(
            embeddim     = self.embeddim , 
            mlpdim       = self.mlpsize , 
            msaDropout   = self.msadropout, 
            mlpDropout   = self.mlpdropout
            ) for _ in range(self.nBlocks)] )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape = self.embeddim),
            nn.Linear(in_features = self.embeddim , out_features = self.nClasses))

    def forward(self, x):
        x = self.patches(x)
        x = self.transformer(x)[:, 0]
        x = self.classifier(x)
        return x