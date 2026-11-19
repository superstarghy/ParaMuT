# Recorded results

Hanyuan Gao and Xiaoxuan Yang license this report and the accompanying CSV compilations under [CC BY 4.0](../licenses/CC-BY-4.0.txt).

The tables report measurements from 112 evaluated checkpoints across six datasets. Each row of the summary gives the training seed, selected epoch, test-set size, clean accuracy, and normalized FGSM and PGD accuracy AUC. The curve tables give the measured accuracy at each perturbation strength, including the clean endpoint. Join the files on `dataset`, `backbone`, `method`, `variant`, and `seed`. [Per-checkpoint summary](summary.csv) · [FGSM and PGD curves](curves.csv) · [Event corruptions](event_corruptions.csv)

All accuracies are percentages of test examples. AUC is the trapezoidal integral of accuracy over the reported strength grid, divided by the width of the interval, also in percent. For multiple seeds, the tables show the arithmetic mean and sample standard deviation of the per-seed metrics. Seeds share a test split. Compare AUCs within the same dataset, backbone, input units, and grid.

## VGG11-BN on images and event frames

| Dataset | Method | Clean (%) | FGSM AUC (%) | PGD-7 AUC (%) |
| --- | --- | ---: | ---: | ---: |
| CIFAR10 | LIF | 94.93 | 44.98 | 16.55 |
| CIFAR10 | PSN-style | 94.77 | 50.48 | 15.23 |
| CIFAR10 | ParaMuT | 95.25 | 49.01 | 18.68 |
| CIFAR100 | LIF | 76.43 | 16.63 | 7.36 |
| CIFAR100 | PSN-style | 76.74 | 18.23 | 7.25 |
| CIFAR100 | ParaMuT | 76.45 | 19.69 | 9.98 |
| CIFAR10-DVS | LIF | 81.20 | 39.18 | 24.21 |
| CIFAR10-DVS | PSN-style | 79.20 | 26.10 | 16.71 |
| CIFAR10-DVS | ParaMuT | 81.20 | 47.20 | 35.43 |
| DVS128 Gesture | LIF | 96.18 | 25.16 | 16.22 |
| DVS128 Gesture | PSN-style | 92.36 | 23.61 | 7.68 |
| DVS128 Gesture | ParaMuT | 96.18 | 51.17 | 41.41 |

CIFAR10 and CIFAR100 used 8 time steps, 200 training epochs, seed 0 and 10,000 test images each. CIFAR10 reports the last epoch. CIFAR100 used the epoch with highest measured test accuracy. Both event datasets used 10 time steps, 100 epochs, seed 1 and test-selected checkpoints. CIFAR10-DVS has 1,000 test recordings and DVS128 Gesture has 288.

The ParaMuT configurations use a causal block matrix `D+B` and a row and column L1 cap on `B`. For CIFAR10 and CIFAR100, the block size is 4, the cap is 0.25, and the fixed diagonal is 1 and 0.5 respectively. The CIFAR10 configuration was selected after examining the sweep. For CIFAR10-DVS, the block size is 5 and the diagonal is learned in [0.5, 1], initialized at 0.75, with a cap of 0.1. For DVS128 Gesture, the block size is 5, the diagonal is 0.25, and the cap is 0.125. PSN-style uses a full-sequence matrix, a fixed threshold and no reset.

The vision sweep used BPDA and seven PGD steps of ε/4 from one random start. FGSM and PGD accuracy count correct predictions on the resulting perturbed inputs. The broad grid spans approximately [0, 8/255] in unit pixel values for CIFAR and [0, 0.01] in normalized frame values for DVS. The CIFAR100 `fgsm_weak_grid` and `pgd_weak_grid` curves contain an independent 11-point evaluation up to approximately 1/255. Only the broad grid contributes to AUC in the summary.

The event-frame curves also report event drop, insertion, polarity flip, frame drop and one-step temporal jitter. Strength denotes a probability for dropping, flipping or moving occupied bins or frames. For insertion, it denotes the requested number of inserted bins relative to the original active-bin count. These corruption curves use strengths of 0.01, 0.03, 0.05 and 0.1, plus the clean endpoint.

## Spikformer v1

| Dataset | Method | Clean (%) | FGSM AUC (%) | PGD-20 AUC (%) |
| --- | --- | ---: | ---: | ---: |
| CIFAR10 | LIF | 94.90 | 79.02 | 47.03 |
| CIFAR10 | PSN-style | 94.45 | 62.97 | 17.90 |
| CIFAR10 | ParaMuT K=2 | 94.30 | 48.76 | 15.69 |
| CIFAR10-DVS | LIF | 80.83 ± 0.64 | 13.39 ± 0.93 | 1.190 ± 0.101 |
| CIFAR10-DVS | PSN-style | 76.50 ± 1.00 | 10.85 ± 0.67 | 1.499 ± 0.128 |
| CIFAR10-DVS | ParaMuT K=4 | 79.50 ± 0.44 | 13.83 ± 0.81 | 1.498 ± 0.176 |
| CIFAR10-DVS | ParaMuT K=8 | 78.63 ± 0.61 | 22.56 ± 1.82 | 2.211 ± 0.172 |

CIFAR10 used 4 time steps, 410 epochs and seed 42. CIFAR10-DVS used 16 time steps, 200 epochs and seeds 2021, 2022 and 2023. Both collected the best checkpoint by test accuracy. The ParaMuT block used a fixed unit diagonal and a 0.25 cap on the row and column L1 norms of the residual.

The evaluation combined attacks from native gradients and gradients through the reset and temporal states. A sample counts as robust if it was correct when clean and survived all attack candidates. PGD-20 used two random restarts per gradient mode, steps of ε/4 and retained successful candidates across iterations and budgets. The CIFAR10 perturbation is shared across time steps in unit pixel space, with AUC over [0, 8/255]. The DVS experiment used continuous nonnegative raw event counts, with AUC over [0, 1]. All seeds were evaluated on the complete test set.

## SHD and UCI-HAR

| Dataset and backbone | Method | Clean (%) | FGSM AUC (%) | PGD-7 AUC (%) |
| --- | --- | ---: | ---: | ---: |
| SHD, DCLS | LIF | 95.04 ± 0.72 | 56.13 ± 5.01 | 31.76 ± 4.42 |
| SHD, DCLS | ParaMuT K=4, uncapped | 95.38 ± 0.39 | 56.72 ± 0.84 | 37.45 ± 0.16 |
| SHD, DCLS | ParaMuT K=16, full-matrix cap 0.75 | 93.61 ± 0.24 | 63.64 ± 2.70 | 50.40 ± 3.13 |
| UCI-HAR, SpikeCNN-BN | LIF | 93.42 | 77.98 | 65.63 |
| UCI-HAR, SpikeCNN-BN | PSN-style | 93.69 | 78.88 | 60.38 |
| UCI-HAR, SpikeCNN-BN | ParaMuT K=4, full-matrix cap 0.75 | 93.99 | 80.20 | 75.49 |
| UCI-HAR, SpikeCNN-BN | ParaMuT K=16, D+B | 93.45 | 77.76 | 73.41 |

**SHD.** The full tables cover ten configurations with DCLS learnable delays and the same ten with zero delay, each at seeds 0, 1 and 2. All use 8,156 training and 2,264 test recordings, 10 ms frames, 140 input channels, variable sequence lengths and 150 epochs. Checkpoint selection used mean test-batch accuracy. Reported accuracy weights each recording equally. The 12-point attack grid spans [0, 0.2] in raw nonnegative event counts, with artificial padding fixed at zero. PGD-7 retained the clean input, FGSM, a single random start and the iterates.

**Historical SHD PSN-style comparison.** The PSN-style measurements come from the separate SpArch comparison with 100 bins, 700 input channels, 128-unit hidden layers and 100 training epochs. The three test-selected checkpoints scored 87.10 ± 1.33% on a fresh draw of initial states. The clean endpoint paired with attack evaluation was 87.17 ± 1.33%, FGSM AUC was 71.49 ± 0.43%, and PGD-7 AUC was 61.63 ± 0.90%. The CSVs record the paired evaluation on all 2,264 samples over eight initial-state draws. Attacks used four draws for gradient estimation and budgets in [0, 0.01] in raw count space. The DCLS comparison measured LIF and ParaMuT only.

**UCI-HAR.** The full tables cover all 22 configurations at seed 0. The subject-disjoint split has 7,352 training windows and 2,947 test windows. Training used 120 epochs and selected the best checkpoint on the test set. Each window contributes equally to reported accuracy. The six-point grid spans [0, 0.2] in units of the training-set standard deviation of each sensor channel. PGD-7 used one random start and ε/4 steps, retaining successful candidates from clean, FGSM, the starting point and each iterate.

For SHD and HAR, `kN` identifies a block of N local time steps. Full-matrix caps constrain row and column L1 norms of the entire causal matrix. The D+B variants use a fixed diagonal of 0.5 and cap the strictly lower triangular residual at 0.25. Test-based checkpoint or configuration selection can bias comparisons on that same test split.
