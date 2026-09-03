import numpy as np

rng = np.random.default_rng(0)
N = 257

print(f"{'offset':>8} {'true var':>12} {'E[x^2]-E[x]^2':>16} {'rel err':>10} "
      f"{'Welford':>12} {'rel err':>10}")
for off in [0, 10, 100, 1000, 10000, 100000]:
    x = (rng.uniform(-2, 2, N) + off).astype(np.float32)
    true = np.var(x.astype(np.float64))

    s = ss = np.float32(0)
    for v in x:
        s = np.float32(s + v)
        ss = np.float32(ss + np.float32(v * v))
    n = np.float32(N)
    naive = np.float32(ss / n) - np.float32(np.float32(s / n) ** 2)

    c = m = m2 = np.float32(0)
    for v in x:
        c = np.float32(c + 1)
        d = np.float32(v - m)
        m = np.float32(m + d / c)
        m2 = np.float32(m2 + d * np.float32(v - m))
    wel = np.float32(m2 / c)

    print(f"{off:>8} {true:>12.6f} {naive:>16.6f} {abs(naive-true)/true:>10.2e} "
          f"{wel:>12.6f} {abs(wel-true)/true:>10.2e}")

print("\nAt offset 1e5 the naive formula returns exactly 0.0, so rsqrt(var+eps)")
print("becomes 1/sqrt(eps) ~ 316 and the activations are silently rescaled by")
print("two orders of magnitude. Welford is unaffected.")
