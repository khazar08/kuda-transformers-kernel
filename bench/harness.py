"""Benchmark harness: CUDA-event timing, hardware peaks, and derived metrics."""
import statistics
import subprocess
import threading

import torch

_CORES_PER_SM = {
    (6, 0): 64, (6, 1): 128, (7, 0): 64, (7, 5): 64,
    (8, 0): 64, (8, 6): 128, (8, 9): 128, (9, 0): 128, (10, 0): 128,
}

_PEAK_BW_GBS = {
    "T4": 320.0, "V100": 900.0, "A100": 1555.0, "H100": 3350.0,
    "L4": 300.0, "L40": 864.0, "RTX 4090": 1008.0, "RTX 3090": 936.0,
    "RTX A6000": 768.0, "A10G": 600.0, "P100": 732.0,
}

def _nvidia_smi(field):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=10,
        )
        return float(out.decode().strip().split("\n")[0])
    except Exception:
        return None

def device_specs():
    """Hardware peaks for the current device, with provenance for each number."""
    props = torch.cuda.get_device_properties(0)
    cc = (props.major, props.minor)
    name = props.name

    cores_per_sm = _CORES_PER_SM.get(cc)
    max_sm_clock_mhz = _nvidia_smi("clocks.max.sm")

    peak_fp32_gflops = None
    if cores_per_sm and max_sm_clock_mhz:
        peak_fp32_gflops = (
            props.multi_processor_count * cores_per_sm * 2 * max_sm_clock_mhz * 1e6
        ) / 1e9

    peak_bw = next((v for k, v in _PEAK_BW_GBS.items() if k in name), None)

    return {
        "name": name,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "sms": props.multi_processor_count,
        "cores_per_sm": cores_per_sm,
        "max_sm_clock_mhz": max_sm_clock_mhz,
        "total_mem_gb": props.total_memory / 1e9,
        "shared_mem_per_block_kb": getattr(props, "shared_memory_per_block", 0) / 1024,
        "max_threads_per_sm": getattr(props, "max_threads_per_multi_processor", None),
        "peak_fp32_gflops": peak_fp32_gflops,
        "peak_bandwidth_gbs": peak_bw,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }

def time_cuda(fn, warmup=25, iters=100):
    """Time `fn` with CUDA events. Returns a dict of ms statistics + clocks."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    clock_before = _nvidia_smi("clocks.current.sm")

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    clock_after = _nvidia_smi("clocks.current.sm")
    times.sort()
    return {
        "mean_ms": statistics.fmean(times),
        "std_ms": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "min_ms": times[0],
        "median_ms": statistics.median(times),
        "p95_ms": times[int(0.95 * (len(times) - 1))],
        "iters": iters,
        "sm_clock_start_mhz": clock_before,
        "sm_clock_end_mhz": clock_after,
    }

def effective_peak_gflops(specs, sm_clock_mhz):
    """Peak FLOP/s at the clock the GPU was ACTUALLY running at."""
    if not (specs.get("cores_per_sm") and sm_clock_mhz):
        return None
    return specs["sms"] * specs["cores_per_sm"] * 2 * sm_clock_mhz * 1e6 / 1e9

class ClockSampler:
    """Sample SM clock continuously in a background thread during timing."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self._handle = None
        self._nvml = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml = None

    def _read(self):
        if self._nvml is not None:
            try:
                return float(self._nvml.nvmlDeviceGetClockInfo(
                    self._handle, self._nvml.NVML_CLOCK_SM))
            except Exception:
                return None
        return _nvidia_smi("clocks.current.sm")

    def _loop(self):
        while not self._stop.is_set():
            v = self._read()
            if v:
                self.samples.append(v)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def stats(self):
        if not self.samples:
            return {}
        return {
            "sm_clock_mean_mhz": statistics.fmean(self.samples),
            "sm_clock_min_mhz": min(self.samples),
            "sm_clock_max_mhz": max(self.samples),
            "sm_clock_samples": len(self.samples),
        }

def time_cuda_roundrobin(named_fns, warmup=10, rounds=30):
    """Time several kernels by INTERLEAVING them, one iteration each per round."""
    for _name, fn in named_fns:
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()

    times = {name: [] for name, _ in named_fns}
    with ClockSampler() as clocks:
        for _ in range(rounds):
            for name, fn in named_fns:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize()
                times[name].append(start.elapsed_time(end))
    clock_stats = clocks.stats()

    out = {}
    for name, samples in times.items():
        samples.sort()
        out[name] = {
            "mean_ms": statistics.fmean(samples),
            "std_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
            "min_ms": samples[0],
            "median_ms": statistics.median(samples),
            "p95_ms": samples[int(0.95 * (len(samples) - 1))],
            "iters": len(samples),
            **clock_stats,
        }
    return out

def gemm_metrics(M, N, K, ms, specs, sm_clock_mhz=None):
    """Derived performance metrics for one GEMM measurement."""
    flops = 2.0 * M * N * K
    seconds = ms * 1e-3
    gflops = flops / seconds / 1e9

    compulsory_bytes = 4.0 * (M * K + K * N + M * N)
    gbs = compulsory_bytes / seconds / 1e9
    intensity = flops / compulsory_bytes

    out = {
        "gflops": gflops,
        "compulsory_gbs": gbs,
        "arithmetic_intensity": intensity,
    }
    if specs.get("peak_fp32_gflops"):
        out["pct_peak_flops"] = 100.0 * gflops / specs["peak_fp32_gflops"]
    if specs.get("peak_bandwidth_gbs"):
        out["pct_peak_bw"] = 100.0 * gbs / specs["peak_bandwidth_gbs"]
    eff = effective_peak_gflops(specs, sm_clock_mhz)
    if eff:
        out["effective_peak_gflops"] = eff
        out["pct_effective_peak"] = 100.0 * gflops / eff
        out["sm_clock_mean_mhz"] = sm_clock_mhz
    return out

def elementwise_metrics(n_elements, bytes_per_element_touched, ms, specs):
    """Derived metrics for memory-bound ops (softmax, LayerNorm)."""
    seconds = ms * 1e-3
    total_bytes = float(n_elements) * bytes_per_element_touched
    gbs = total_bytes / seconds / 1e9
    out = {"achieved_gbs": gbs}
    if specs.get("peak_bandwidth_gbs"):
        out["pct_peak_bw"] = 100.0 * gbs / specs["peak_bandwidth_gbs"]
    return out

def print_specs(specs):
    print(f"GPU                : {specs['name']}  (sm_{specs['compute_capability'].replace('.','')})")
    print(f"SMs                : {specs['sms']}  x {specs['cores_per_sm']} FP32 cores")
    print(f"Max SM clock       : {specs['max_sm_clock_mhz']} MHz")
    print(f"Peak FP32          : {specs['peak_fp32_gflops']:.0f} GFLOP/s"
          if specs["peak_fp32_gflops"] else "Peak FP32          : unknown")
    print(f"Peak bandwidth     : {specs['peak_bandwidth_gbs']} GB/s"
          if specs["peak_bandwidth_gbs"] else "Peak bandwidth     : unknown")
    if specs["peak_fp32_gflops"] and specs["peak_bandwidth_gbs"]:
        ridge = specs["peak_fp32_gflops"] / specs["peak_bandwidth_gbs"]
        print(f"Roofline ridge     : {ridge:.1f} flops/byte")
    print(f"torch / CUDA       : {specs['torch']} / {specs['cuda']}")
