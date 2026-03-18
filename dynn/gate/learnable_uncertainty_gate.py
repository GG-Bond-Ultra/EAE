from typing import Tuple
import torch
from torch import Tensor
from dynn.gate.gate import Gate, GateType
from metrics_utils import compute_detached_uncertainty_metrics

class LearnableUncGate(Gate):
    def __init__(self):
        super(Gate, self).__init__()
        self.gate_type = GateType.UNCERTAINTY
        self.dim = 4
        self.linear = torch.nn.Linear(self.dim, 1)

    def forward(self, logits: Tensor) -> Tensor:
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)  
        top2probs, top2indices = torch.topk(probs, 2, dim=1)
        p_max, next_p_max = top2probs[:, 0], top2probs[:, 1]
        margins = p_max - next_p_max
        entropy = -torch.sum(probs * log_probs, dim=1)
        logits_scaled = logits * 2
        pow_probs = torch.nn.functional.softmax(logits_scaled, dim=1)  
        log_pow_probs = torch.nn.functional.log_softmax(logits_scaled, dim=1) 
        entropy_pow = -torch.sum(pow_probs * log_pow_probs, dim=1) 
        uncertainty_metrics = torch.stack([p_max, entropy, margins, entropy_pow], dim=1)

        return self.linear(uncertainty_metrics)


    def get_flops(self, num_classes):
        # compute flops for preprocssing of input and then for linear layer.
        p_max_flops = num_classes # comparison across the logits
        margin_flops = num_classes + 1 # compare top1 with top2
        entropy_flops = num_classes * 2 # compute entropy p log p then sum those values
        entropy_pow_flops = num_classes * 5 # 1 for raising to power, 1 for computing normalizing denom, 1 for scaling each pow then 2 for entropy computation
        linear_flops = self.dim + 1 # dim + bias
        return p_max_flops + margin_flops + entropy_flops + entropy_pow_flops + linear_flops
