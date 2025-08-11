
import torch

def mineTriplets(embeddings , labels):
    pairwise = torch.cdist(embeddings, embeddings , p = 2)
    triplets = []
    for i , label in enumerate(labels):
        posMask = (labels == label) & torch.arange(len(labels), device=labels.device) != i
        negMask = (labels != label)
        if posMask.sum() == 0 or negMask.sum() == 0:
            continue
        hardestNegative = pairwise[i][negMask].argmin()
        hardestPositive = pairwise[i][posMask].argmax()
        hardestPositive = posMask.nonzero()[hardestPositive].item()
        hardestNegative = negMask.nonzero()[hardestNegative].item()
        triplets.append((i , hardestPositive, hardestNegative))
    return triplets