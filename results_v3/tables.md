
## baf  (runs per method: 227, kappa=0.1, t=8, alpha=0.0001)

| Method | Mean recall | False flags /100k | Top-1 loc (unpaired) | Correctly localized share |
| --- | --- | --- | --- | --- |
| schema | 0.000 | 0 | nan | 0.000 |
| minmax | 0.443 | 36 | 0.811 | 0.443 |
| autoprofile | 0.538 | 36 | 0.881 | 0.537 |
| robust_z | 0.159 | 5 | 0.732 | 0.159 |
| iforest | 0.005 | 22 | 0.000 | 0.000 |
| pca_recon | 0.313 | 10 | 0.337 | 0.128 |
| conformance | 0.312 | 6 | 0.872 | 0.311 |
| contracts_domain | 0.531 | 0 | 1.000 | 0.531 |
| contracts | 0.531 | 0 | 0.955 | 0.531 |

Recall by fault type (contracts / autoprofile / minmax):
- clipping: 0.00 / 0.00 / 0.00
- default_fill: 0.55 / 0.55 / 0.48
- encoding: 0.90 / 0.96 / 0.96
- inconsistency: 0.65 / 0.65 / 0.00
- magnitude: 1.00 / 1.00 / 1.00
- misjoin: 0.00 / 0.00 / 0.00
- scale: 0.96 / 0.96 / 0.96
- stale: 0.00 / 0.00 / 0.00

RQ2 relation checks vs domain only: mean +0.000; wins 7, losses 0, ties 41 over 48 cells
Wilcoxon over cells, contracts minus baseline (Holm-adjusted p): minmax +0.083 (p=0.485); autoprofile -0.008 (p=0.000703); robust_z +0.355 (p=4.4e-05); iforest +0.501 (p=5.25e-06); pca_recon +0.210 (p=0.000145); conformance +0.211 (p=0.000145)

Bootstrap 95% intervals over runs:
- Recall, contracts - autoprofile: -0.007 [-0.018, -0.000]
- Recall, contracts - conformance: +0.219 [+0.169, +0.272]
- Correctly localized, contracts - minmax: +0.088 [+0.050, +0.128]
- Correctly localized, contracts - conformance: +0.220 [+0.170, +0.273]

| Paging variant | Events detected | Median delay | False pages /100k |
| --- | --- | --- | --- |
| autoprofile|perfeature | 0.56 | 17 | 2.2 |
| autoprofile|pooled | 0.56 | 16 | 5.0 |
| conformance|pooled | 0.33 | 32 | 0.0 |
| contracts_ds|burst | 0.65 | 38 | 0.5 |
| contracts_ds|hard | 0.63 | 37 | 0.5 |
| contracts|all | 0.66 | 36 | 0.5 |
| contracts|all_A1000 | 0.66 | 0 | 0.5 |
| contracts|all_A10000 | 0.66 | 14 | 0.5 |
| contracts|burst | 0.66 | 38 | 0.5 |
| contracts|cusum | 0.66 | 61 | 0.5 |
| contracts|every | 0.56 | 0 | 0.2 |
| contracts|hard | 0.64 | 36 | 0.5 |
| contracts|hard_A1000 | 0.64 | 0 | 0.5 |
| contracts|hard_A10000 | 0.64 | 14 | 0.5 |
| contracts|pooled | 0.56 | 22 | 0.0 |
| contracts|rates | 0.18 | 499 | 0.5 |
| iforest|pooled | 0.07 | 534 | 0.0 |
| minmax|pooled | 0.45 | 16 | 5.0 |
| pca_recon|pooled | 0.34 | 29 | 0.0 |
| psi | 0.29 | 999 | 462.7 |
| robust_z|pooled | 0.21 | 45 | 0.0 |

Event detection, contracts(all) - autoprofile: +0.106 [+0.066, +0.145]
Detection by prevalence (contracts|all): 0.01: 0.56, 0.1: 0.63, 1: 0.80
Detection by fault (contracts|all): clipping: 0.15, default_fill: 0.72, encoding: 1.00, inconsistency: 0.73, magnitude: 1.00, misjoin: 0.00, scale: 1.00, stale: 0.52

Calibration: budget 10/100k, realized contracts 0 (ratio 0.03); flag bursts 1; top features [['device_distinct_emails_8w', 1]]