from sklearn.metrics import average_precision_score
import torch.nn.functional as F

def rankK(outputs, labels, ks = [1, 5]):
    k = max(ks)
    _, pred = outputs.topk(k, dim = 1, largest=True, sorted=True)  
    pred = pred.t() 
    correct = pred.eq(labels.view(1, -1).expand_as(pred)) 
    results = {}
    for k in ks:
        results[f"Rank-{k}"] = correct[:k].reshape(-1).float().sum(0).item() / labels.size(0)
    return results


def mAP(outputs, labels, nClasses):
    probs = F.softmax(outputs, dim = 1).cpu().numpy()
    labelsOnehot = F.one_hot(labels, num_classes = nClasses).cpu().numpy()
    APs = []
    for c in range(nClasses):
        if labelsOnehot[:, c].sum() == 0:
            continue
        APs.append(average_precision_score(labelsOnehot[:, c], probs[:, c]))
    return sum(APs) / len(APs) if APs else 0.0
