import torch
from torch import nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, nChannels, patchSize, embedDim, nPatches , *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.convolve        = nn.Conv2d(in_channels = nChannels , out_channels = embedDim, kernel_size = patchSize, stride = patchSize)
        self.flatten         = nn.Flatten(start_dim = 1 , end_dim = 2)
        self.tokenEmbeddings = nn.Parameter(torch.rand((1 , 1 , embedDim), requires_grad = True))
        self.posEmbeddings   = nn.Parameter(torch.rand((1 , nPatches + 1 , embedDim), requires_grad = True))
    
    def forward(self, x):
        x = self.convolve(x)
        x = x.permute((0 , 2 , 3 , 1))
        x = self.flatten(x)
        b = x.shape[0]
        x = torch.cat((self.tokenEmbeddings.expand(b, -1 ,-1) , x), dim = 1)
        x = x + self.posEmbeddings
        return x
    
class MSABlock(nn.Module):
    def __init__(self, embedDim, nHeads = 12 , dropout = 0.1 , *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm      = nn.LayerNorm(normalized_shape = embedDim)
        self.attention = nn.MultiheadAttention(num_heads = nHeads , dropout = dropout , embed_dim = embedDim , batch_first = True)

    def forward(self, x):
        x = self.norm(x)
        x , _ = self.attention(query = x, key = x , value = x, need_weights = False)
        return x
    
class MLPBlock(nn.Module):
    def __init__(self, inDim , interDim, outDim , dropout = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = nn.LayerNorm(normalized_shape = inDim)
        self.mlp    = nn.Sequential(
            nn.Linear(in_features = inDim , out_features = interDim),
            nn.GELU(),
            nn.Dropout(p = dropout),
            nn.Linear(in_features = interDim, out_features = outDim),
            nn.Dropout(p = dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.mlp(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, embedDim , mlpDim , mlpDropout = 0.1, msaDropout = 0.1 , nHeads = 12 , *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp = MLPBlock(inDim = embedDim , outDim = embedDim , interDim = mlpDim, dropout = mlpDropout)
        self.msa = MSABlock(embedDim = embedDim, nHeads = nHeads, dropout = msaDropout)

    def forward(self, x):
        x = self.msa(x) + x
        x = self.mlp(x) + x
        return x
    
class VIT(nn.Module):
    def __init__(self, 
                 imageHeight , 
                 imageWidth , 
                 nClasses ,
                 featureDim = 512, 
                 nBlocks    = 12,
                 patchSize  = 16, 
                 nChannels   = 3, 
                 mlpDropout = 0.1 , 
                 msaDropout = 0.0 ,
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.imageHeight = imageHeight
        self.imageWidth  = imageWidth
        self.nClasses    = nClasses
        self.nBlocks     = nBlocks
        self.patchSize   = patchSize
        self.nChannels   = nChannels
        self.embedDim    = int(self.nChannels * (self.patchSize ** 2))
        self.nPatches    = int((self.imageHeight * self.imageWidth) / (self.patchSize ** 2))
        self.mlpdropout  = mlpDropout
        self.msadropout  = msaDropout
        self.mlpsize     = (self.patchSize ** 2) * 12
        self.featureDim  =  featureDim

        self.patches     = PatchEmbedding(
            nChannels    = self.nChannels , 
            patchSize    = patchSize, 
            nPatches     = self.nPatches, 
            embedDim     = self.embedDim)
        
        self.transformer = nn.Sequential(*[TransformerBlock(
            embedDim     = self.embedDim , 
            mlpDim       = self.mlpsize , 
            msaDropout   = msaDropout, 
            mlpDropout   = mlpDropout
            ) for _ in range(self.nBlocks)] )

        self.featureHead = nn.Sequential(
            nn.LayerNorm(normalized_shape = self.embedDim),
            nn.Linear(in_features = self.embedDim, out_features = featureDim),
            nn.BatchNorm1d(featureDim),
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape = self.embedDim),
            nn.Linear(in_features = self.embedDim , out_features = nClasses),
            )

    def forward(self, x):
        x = self.patches(x)
        x = self.transformer(x)[:, 0]
        features = self.featureHead(x)
        features = F.normalize(features)
        logits = self.classifier(features)
        return logits, features