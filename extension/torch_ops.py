"""Swap the custom kernels into a real model's inference path."""
import torch
import torch.nn as nn

from extension.build import load_extension

class FastLinear(nn.Module):
    """Drop-in nn.Linear replacement backed by our SGEMM."""

    def __init__(self, linear: nn.Linear, stage: str = "vectorized",
                 m_threshold: int = 128):
        super().__init__()
        assert linear.weight.dtype == torch.float32, "kernels are fp32-only"
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.stage = stage
        self.m_threshold = m_threshold
        self.register_buffer(
            "weight_t", linear.weight.detach().t().contiguous(), persistent=False
        )
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone(),
                                 persistent=False)
        else:
            self.bias = None
        self._ext = load_extension()

    def forward(self, x):
        M = x.numel() // x.shape[-1]
        if M < self.m_threshold:
            return torch.nn.functional.linear(
                x, self.weight_t.t(), self.bias)

        y = self._ext.sgemm(x.contiguous(), self.weight_t, self.stage)
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self):
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, stage={self.stage}, "
                f"m_threshold={self.m_threshold}")

class FastConv1D(nn.Module):
    """Drop-in replacement for HuggingFace's Conv1D, backed by our SGEMM."""

    def __init__(self, conv1d, stage: str = "vectorized", m_threshold: int = 128):
        super().__init__()
        assert conv1d.weight.dtype == torch.float32, "kernels are fp32-only"
        self.in_features = conv1d.weight.shape[0]
        self.out_features = conv1d.weight.shape[1]
        self.stage = stage
        self.m_threshold = m_threshold
        self.register_buffer("weight", conv1d.weight.detach().contiguous(),
                             persistent=False)
        if getattr(conv1d, "bias", None) is not None:
            self.register_buffer("bias", conv1d.bias.detach().clone(),
                                 persistent=False)
        else:
            self.bias = None
        self._ext = load_extension()

    def forward(self, x):
        M = x.numel() // x.shape[-1]
        if M < self.m_threshold:
            out = torch.matmul(x, self.weight)
            return out + self.bias if self.bias is not None else out
        y = self._ext.sgemm(x.contiguous(), self.weight, self.stage)
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self):
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, stage={self.stage}")

class FastLayerNorm(nn.Module):
    """Drop-in nn.LayerNorm replacement backed by our fused kernel."""

    def __init__(self, ln: nn.LayerNorm):
        super().__init__()
        assert len(ln.normalized_shape) == 1, \
            "only last-dim LayerNorm is supported"
        self.eps = ln.eps
        self.register_buffer("gamma", ln.weight.detach().contiguous()
                             if ln.weight is not None else None, persistent=False)
        self.register_buffer("beta", ln.bias.detach().contiguous()
                             if ln.bias is not None else None, persistent=False)
        self._ext = load_extension()

    def forward(self, x):
        return self._ext.layernorm(x.contiguous(), self.gamma, self.beta, self.eps)

def patch_model(model, stage="vectorized", min_features=256, m_threshold=128,
                patch_linear=True, patch_layernorm=True, verbose=True):
    """Recursively replace Linear / LayerNorm modules. Returns a report dict."""
    replaced = {"linear": 0, "conv1d": 0, "layernorm": 0,
                "skipped_small": 0, "skipped_dtype": 0}

    try:
        from transformers.pytorch_utils import Conv1D as HFConv1D
    except Exception:
        HFConv1D = ()

    def recurse(module):
        for name, child in list(module.named_children()):
            if patch_linear and HFConv1D and isinstance(child, HFConv1D):
                k, n = child.weight.shape
                if child.weight.dtype != torch.float32:
                    replaced["skipped_dtype"] += 1
                elif min(k, n) < min_features:
                    replaced["skipped_small"] += 1
                else:
                    setattr(module, name, FastConv1D(child, stage=stage,
                                                     m_threshold=m_threshold))
                    replaced["conv1d"] += 1
            elif patch_linear and isinstance(child, nn.Linear):
                if child.weight.dtype != torch.float32:
                    replaced["skipped_dtype"] += 1
                elif min(child.in_features, child.out_features) < min_features:
                    replaced["skipped_small"] += 1
                else:
                    setattr(module, name, FastLinear(child, stage=stage,
                                                     m_threshold=m_threshold))
                    replaced["linear"] += 1
            elif patch_layernorm and isinstance(child, nn.LayerNorm):
                if child.weight is not None and child.weight.dtype != torch.float32:
                    replaced["skipped_dtype"] += 1
                elif len(child.normalized_shape) != 1:
                    replaced["skipped_small"] += 1
                else:
                    setattr(module, name, FastLayerNorm(child))
                    replaced["layernorm"] += 1
            else:
                recurse(child)

    recurse(model)
    total_matmul = replaced["linear"] + replaced["conv1d"]
    if verbose:
        print(f"patched {replaced['linear']} nn.Linear + {replaced['conv1d']} "
              f"HF Conv1D = {total_matmul} matmul layers (stage={stage})")
        print(f"patched {replaced['layernorm']} LayerNorm -> FastLayerNorm")
        print(f"skipped {replaced['skipped_small']} (below min_features={min_features}), "
              f"{replaced['skipped_dtype']} (non-fp32)")
        if total_matmul == 0:
            print("WARNING: no matmul layers were patched. This model's linear "
                  "layers are neither nn.Linear nor Conv1D, so the end-to-end "
                  "measurement would be meaningless.")
    return replaced

def assert_logits_parity(model_a, model_b, input_ids, rtol=2e-3, atol=2e-3):
    """Confirm patching did not change what the model computes."""
    with torch.no_grad():
        a = model_a(input_ids).logits.float()
        b = model_b(input_ids).logits.float()
    max_abs = (a - b).abs().max().item()
    same_argmax = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    print(f"logits max abs diff : {max_abs:.3e}")
    print(f"argmax token match  : {100*same_argmax:.2f}%")
    assert same_argmax == 1.0, "patched model predicts different tokens"
    return {"max_abs_diff": max_abs, "argmax_match": same_argmax}
