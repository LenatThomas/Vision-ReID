from sklearn.metrics import average_precision_score
import torch.nn.functional as F


def rankK(predictions , targets, ks = [1 , 2 , 3 , 5]):
    ks = sorted(ks)
    kHits = {k : 0 for k in ks}
    nQueries = len(predictions)
    for i in range(nQueries):
        p = predictions[i]
        t = targets[i]
        for k in ks:
            if t in p[:k]:
                kHits[k] += 1
    rankAccuracy = {k: kHits[k] / nQueries for k in ks}
    return rankAccuracy
    
def averagePrecision(predictions , target):
    precision   = []
    relevant    = 0
    for i , p in enumerate(predictions, start = 1):
        if p == target:
            relevant += 1
            precision.append(relevant / i)
    return sum(precision) / len(precision) if precision else 0.0


def mAP(predictions , targets):
    precisions = [averagePrecision(predictions[i] , targets[i]) for i in range(len(predictions))]
    return sum(precisions) / len(precisions)