"""End-to-end: tokens/sec through a real transformer, before vs after patching."""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.harness import device_specs, print_specs, time_cuda
from extension.torch_ops import assert_logits_parity, patch_model

def load_model(model_id, lora_id=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 torch_dtype=torch.float32)
    if lora_id:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_id)
        model = model.merge_and_unload()
        print(f"merged LoRA adapter {lora_id} into base weights")
    return model.cuda().eval(), tok

@torch.no_grad()
def bench_prefill(model, input_ids, warmup=5, iters=20):
    fn = lambda: model(input_ids)
    t = time_cuda(fn, warmup=warmup, iters=iters)
    n_tokens = input_ids.numel()
    return t, n_tokens / (t["mean_ms"] * 1e-3)

@torch.no_grad()
def bench_decode(model, input_ids, new_tokens=32, warmup=2, iters=5):
    def fn():
        model.generate(input_ids, max_new_tokens=new_tokens, do_sample=False,
                       use_cache=True, pad_token_id=0)
    t = time_cuda(fn, warmup=warmup, iters=iters)
    return t, new_tokens / (t["mean_ms"] * 1e-3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2",
                    help="any HF causal LM; use your own fine-tuned checkpoint")
    ap.add_argument("--lora", default=None, help="optional peft adapter id/path")
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32],
                    help="decode batch sizes; M = batch during decode")
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--stage", default="vectorized")
    args = ap.parse_args()

    specs = device_specs()
    print_specs(specs)

    base, tok = load_model(args.model, args.lora)
    ids = torch.randint(0, tok.vocab_size, (1, args.prompt_len), device="cuda")

    print(f"\n=== PREFILL, prompt_len={args.prompt_len} (M={args.prompt_len}) ===")
    t_base, tps_base = bench_prefill(base, ids)

    import copy
    patched = copy.deepcopy(base)
    report = patch_model(patched, stage=args.stage)
    print()
    assert_logits_parity(base, patched, ids)

    t_ours, tps_ours = bench_prefill(patched, ids)
    print(f"\n{'impl':<20} {'ms':>10} {'±ms':>8} {'tokens/sec':>12} {'speedup':>9}")
    print(f"{'torch (cuBLAS)':<20} {t_base['mean_ms']:>10.3f} "
          f"{t_base['std_ms']:>8.3f} {tps_base:>12,.0f} {'1.00x':>9}")
    print(f"{'ours ('+args.stage+')':<20} {t_ours['mean_ms']:>10.3f} "
          f"{t_ours['std_ms']:>8.3f} {tps_ours:>12,.0f} "
          f"{tps_ours/tps_base:>8.2f}x")

    for bs in args.batch:
        bids = torch.randint(0, tok.vocab_size, (bs, 32), device="cuda")
        print(f"\n=== DECODE, batch={bs} (M={bs} per step), "
              f"{args.new_tokens} new tokens ===")
        tb, tpsb = bench_decode(base, bids, args.new_tokens)
        to, tpso = bench_decode(patched, bids, args.new_tokens)
        note = ("  <- below m_threshold, FastLinear routes to torch"
                if bs < 128 else "")
        print(f"{'torch (cuBLAS)':<20} {tb['mean_ms']:>10.2f} ms  "
              f"{tpsb:>8.1f} steps/s")
        print(f"{'ours ('+args.stage+')':<20} {to['mean_ms']:>10.2f} ms  "
              f"{tpso:>8.1f} steps/s   {tpso/tpsb:.2f}x{note}")

if __name__ == "__main__":
    main()
