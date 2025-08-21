
import torch

def mineTriplets(anchors , anchorLabels , pools , poolLabels):
    triplets = {
            'anchors' : [],
            'positives' : [],
            'negatives' : []
    }
    pairwise = torch.cdist(anchors, pools, p = 2 )
    anchorLabels = torch.as_tensor(anchorLabels, dtype=torch.long)
    poolLabels   = torch.as_tensor(poolLabels, dtype=torch.long)
    for i , (anchor , label) in enumerate(zip(anchors, anchorLabels)):
        posMask = (poolLabels == label)
        negMask = (poolLabels != label)
        if posMask.sum() == 0 or negMask.sum() == 0:
            continue
        hardestNegative = pairwise[i][negMask].argmin()
        hardestNegative = torch.nonzero(negMask, as_tuple=False)[hardestNegative]
        hardestPositive = pairwise[i][posMask].argmax()
        hardestPositive = torch.nonzero(posMask, as_tuple=False)[hardestPositive]
        triplets['anchors'].append(anchor)
        triplets['positives'].append(pools[hardestPositive])
        triplets['negatives'].append(pools[hardestNegative])

    if len(triplets['anchors']) > 0:
        triplets = {k: torch.stack(v) for k, v in triplets.items()}
    else:
        triplets = {k: torch.empty(0) for k in triplets}
    return triplets
