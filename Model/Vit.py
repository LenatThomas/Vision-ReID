import torch
from torch import nn
import torch.nn.functional as F
from Modules import PatchEmbedding, TransformerBlock
    
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
            nn.LayerNorm(normalized_shape = featureDim),
            nn.Linear(in_features = featureDim , out_features = nClasses),
            )

    def forward(self, x):
        x = self.patches(x)
        x = self.transformer(x)[:, 0]
        features = self.featureHead(x)
        features = F.normalize(features)
        logits = self.classifier(features)
        return logits, features
    
class VITEmbedding(nn.Module):
    def __init__(self, baseModel):
        super().__init__()
        self.patches = baseModel.patches
        self.transformer = baseModel.transformer
        self.featureHead = baseModel.featureHead

    def forward(self, x):
        x = self.patches(x)
        x = self.transformer(x)[:, 0]
        features = self.featureHead(x)
        features = F.normalize(features)
        return features