from calibration import calibration_curve
import torch 
from utils import free 
import scipy
import numpy as np

def compute_entropy(probs):
    # 使用 PyTorch 的张量操作计算熵
    log_probs = torch.log(probs + 1e-10)  # 加上一个小的常数以避免 log(0)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy
# @profile
def compute_detached_uncertainty_metrics(logits, targets):
    
    probs = torch.nn.functional.softmax(logits, dim=1)

    top2prob, _ = torch.topk(probs, 2)

    p_max = top2prob[:,0]
    # next_p_max = top2prob[:,1]
    margins = p_max-top2prob[:,1]

    # p_max = free(p_max)
    # margins = free(margins)

    entropy = scipy.stats.entropy(free(probs), axis=1)

    pow_probs = probs**2
    pow_probs = pow_probs / pow_probs.sum(dim=1, keepdim=True)
    entropy_pow = scipy.stats.entropy(free(pow_probs), axis=1)

    # p_max = p_max
    # margins = margins
    # entropy = compute_entropy(probs)

    # pow_probs = probs**2
    # pow_probs = pow_probs / pow_probs.sum(dim=1, keepdim=True)
    # entropy_pow = compute_entropy(pow_probs)
    

    ece = -1
    # if targets is not None:
    #     _, predicted = logits.max(1)
    #     correct = predicted.eq(targets)
    #     ground_truth = free(correct)
    #     _, _, ece = calibration_curve(ground_truth, p_max,strategy='quantile',n_bins=15)
    
    
    return list(p_max), list(entropy), ece, list(margins), list(entropy_pow)

def compute_detached_score(logits, targets): # score value to obtain conformal intervals
    probs = torch.nn.functional.softmax(logits, dim=1)
    s = 1-probs[np.arange(logits.shape[0]),free(targets)]
    return list(free(s))
    
    