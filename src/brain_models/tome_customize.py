import torch
import torch.nn as nn
from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
from tome.utils import parse_r

class TokenMerging(nn.Module):
    def __init__(self, r=0, trace_source=False, prop_attn=True):
        super().__init__()
        self.r = r
        self.trace_source = trace_source
        self.prop_attn = prop_attn
        self._tome_info = {
            "r": self.r,
            "size": None,
            "source": None,
            "trace_source": self.trace_source,
            "prop_attn": self.prop_attn,
            "class_token": False,  # Set this based on your specific model's requirements
            "distill_token": False,  # Set this based on your specific model's requirements
        }

    def forward(self, x: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
        # Reset tome_info for each batch, if needed
        self._tome_info["source"] = None
        self._tome_info["size"] = None

        r = self._tome_info["r"]
        
        if r > 0:
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self.trace_source:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            try:
                x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])
            except Exception as e:
                print(f"Error in merge_wavg: {str(e)}")
                import pdb; pdb.set_trace()
        return x
