
## ieee  (runs per method: 228, kappa=0.1, t=80, alpha=0.001)

| Method | Mean recall | False flags /100k | Top-1 loc (unpaired) | Correctly localized share |
| --- | --- | --- | --- | --- |
| schema | 0.000 | 0 | nan | 0.000 |
| minmax | 0.426 | 3219 | 0.482 | 0.413 |
| autoprofile | 0.520 | 3530 | 0.587 | 0.506 |
| robust_z | 0.258 | 472 | 0.420 | 0.254 |
| iforest | 0.003 | 765 | 0.000 | 0.000 |
| pca_recon | 0.269 | 822 | 0.452 | 0.244 |
| conformance | 0.312 | 820 | 0.502 | 0.285 |
| contracts_domain | 0.503 | 2073 | 0.600 | 0.496 |
| contracts | 0.517 | 2173 | 0.634 | 0.506 |

Recall by fault type (contracts / autoprofile / minmax):
- clipping: 0.04 / 0.02 / 0.01
- default_fill: 0.32 / 0.30 / 0.29
- encoding: 0.95 / 1.00 / 1.00
- inconsistency: 0.72 / 0.72 / 0.02
- magnitude: 1.00 / 1.00 / 1.00
- misjoin: 0.01 / 0.03 / 0.02
- scale: 0.95 / 0.94 / 0.94
- stale: 0.02 / 0.03 / 0.03

RQ2 relation checks vs domain only: mean +0.014; wins 25, losses 0, ties 23 over 48 cells
Wilcoxon over cells, contracts minus baseline (Holm-adjusted p): minmax +0.087 (p=0.902); autoprofile -0.002 (p=0.014); robust_z +0.252 (p=1.78e-07); iforest +0.497 (p=1.44e-08); pca_recon +0.241 (p=2.1e-07); conformance +0.197 (p=0.0652)

Bootstrap 95% intervals over runs:
- Recall, contracts - autoprofile: -0.003 [-0.019, +0.014]
- Recall, contracts - conformance: +0.205 [+0.155, +0.262]
- Correctly localized, contracts - minmax: +0.093 [+0.054, +0.136]
- Correctly localized, contracts - conformance: +0.221 [+0.171, +0.276]

| Paging variant | Events detected | Median delay | False pages /100k |
| --- | --- | --- | --- |
| autoprofile|pooled | 0.51 | 2 | 974.5 |
| conformance|pooled | 0.49 | 237 | 96.9 |
| contracts|all | 0.59 | 78 | 4619.1 |
| contracts|all_A1000 | 0.63 | 18 | 7711.6 |
| contracts|all_A10000 | 0.63 | 54 | 5952.2 |
| contracts|cusum | 0.59 | 139 | 4530.2 |
| contracts|every | 0.66 | 0 | 22540.4 |
| contracts|hard | 0.54 | 78 | 4266.9 |
| contracts|hard_A1000 | 0.58 | 17 | 7051.6 |
| contracts|hard_A10000 | 0.57 | 42 | 5419.6 |
| contracts|rates | 0.07 | 498 | 294.2 |
| iforest|pooled | 0.07 | 1997 | 88.9 |
| minmax|pooled | 0.41 | 6 | 891.1 |
| pca_recon|pooled | 0.30 | 85 | 99.1 |
| psi | 0.35 | 999 | 1887.7 |
| robust_z|pooled | 0.25 | 42 | 41.9 |

Event detection, contracts(all) - autoprofile: +0.079 [+0.035, +0.123]
Detection by prevalence (contracts|all): 0.01: 0.44, 0.1: 0.55, 1: 0.76
Detection by fault (contracts|all): clipping: 0.26, default_fill: 0.45, encoding: 0.90, inconsistency: 0.77, magnitude: 1.00, misjoin: 0.07, scale: 0.90, stale: 0.28

Calibration: budget 100/100k, realized contracts 2173 (ratio 21.73); flag bursts 4170; top features [['D11', 2388], ['V323', 1370], ['V324', 1362], ['V330', 1357], ['V329', 1350]]