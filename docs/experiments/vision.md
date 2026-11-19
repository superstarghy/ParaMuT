# Vision experiments

The VGG11-BN comparison covers all four vision datasets. A separate Spikformer v1 comparison covers CIFAR10 and CIFAR10-DVS. Both use a fixed-threshold, full-sequence mixing matrix for the PSN-style control. The two backbones use different neuron dynamics, preprocessing and attack protocols. [Recorded results](../../results/README.md)

## Dataset roots

Point each command at one dataset root with `--data-dir`. For CIFAR10 and CIFAR100, pass `--download` on the first training run to obtain the official torchvision split. CIFAR10-DVS uses the SpikingJelly dataset layout. The first download or conversion builds a frame cache for the requested number of timesteps. For manually downloaded DVS128 Gesture, place the four distribution files (`DvsGesture.tar.gz`, `gesture_mapping.csv`, `LICENSE.txt`, and `README.txt`) in `data/dvs128gesture/download/` and run without `--download`. A prepared `frames_number_10_split_by_number/` cache with `train/` and `test/` subdirectories also works. Dataset licenses and download conditions apply separately from the code. [IBM DVS Gesture download](https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794)

| Dataset | CLI value | Train / test | VGG input | Split |
| --- | --- | ---: | --- | --- |
| CIFAR10 | `cifar10` | 50,000 / 10,000 | RGB, 32 × 32 | Official |
| CIFAR100 | `cifar100` | 50,000 / 10,000 | RGB, 32 × 32 | Official |
| CIFAR10-DVS | `cifar10dvs` | 9,000 / 1,000 | 10 frames, 48 × 48 | Class-stratified 90/10, split seed 0 |
| DVS128 Gesture | `dvs128gesture` | 1,176 / 288 | 10 frames, 128 × 128 | Official subject split |

## VGG11-BN

Select `lif`, `paramut`, or `psn`. Training saves epoch-level metrics in `history.csv` and the selected weights in `model.pt`. Run each trial in a fresh output directory. The evaluator reports one row per method and strength in `robustness.csv`, with accuracy as a fraction. By default it scores every sample in the checkpoint's configured test set.

```bash
python run_vision.py train --dataset cifar10 --method paramut \
  --data-dir data/cifar10 --output-dir outputs/cifar10/paramut --download
python run_vision.py evaluate --checkpoint outputs/cifar10/paramut/model.pt \
  --data-dir data/cifar10

python run_vision.py train --dataset dvs128gesture --method psn \
  --data-dir data/dvs128gesture --output-dir outputs/dvs128gesture/psn
python run_vision.py evaluate --checkpoint outputs/dvs128gesture/psn/model.pt \
  --data-dir data/dvs128gesture --event-corruptions
```

| Dataset | Training seed | T / ParaMuT K | ParaMuT diagonal D | Residual B cap | Epochs / batch | Checkpoint |
| --- | ---: | --- | --- | ---: | --- | --- |
| CIFAR10 | 0 | 8 / 4 | 1 | 0.25 | 200 / 128 | Last, test every 50 epochs |
| CIFAR100 | 0 | 8 / 4 | 0.5 | 0.25 | 200 / 128 | Best test score every 20 epochs |
| CIFAR10-DVS | 1 | 10 / 5 | Learned in [0.5, 1], initialized at 0.75 | 0.1 | 100 / 16 | Best test score every 10 epochs |
| DVS128 Gesture | 1 | 10 / 5 | 0.25 | 0.125 | 100 / 8 | Best test score every 10 epochs |

ParaMuT mixes each block through a causal matrix with diagonal D and strictly lower triangular residual B. Row and column L1 caps apply to B. State carries across block boundaries with a soft reset. PSN-style uses a full matrix over all T steps with no boundary reset. All VGG cells use threshold 1, time constant 1.1 and a triangular surrogate. Static images are repeated over T. Event frames preserve both polarities and are divided by the largest count in the sample (denominator at least one), then resized.

The VGG runs use FP32, SGD with momentum 0.9, initial learning rate 0.1 and cosine decay to 0.00001. The temporal loss averages cross-entropy and a squared error to a target logit of one, weighted 0.95 and 0.05. Static runs use weight decay 0.00005, crop and flip, CIFAR AutoAugment, channel normalization and random erasing. Event runs use weight decay 0.0001, dropout 0.1, label smoothing 0.05, spatial crop and random erasing. CIFAR10's reported ParaMuT diagonal was chosen in an exploratory sweep. The best-checkpoint policies use the test split for selection.

VGG FGSM and PGD perturb pixels before channel normalization, or the normalized event frames in [0, 1]. The main grid uses approximately 1, 2, 4 and 8 divided by 255 for CIFAR. The DVS grid is 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.0075 and 0.01. PGD uses seven steps, step size ε/4, one random start and final-iterate accuracy. The direct encoder passes its input gradient through. Original PGD evaluations consumed random starts in checkpoint evaluation order. The public evaluator sets a repeatable seed for each invocation, so small differences are possible. Optional DVS corruptions measure event drop, insertion, polarity flip, frame drop and local temporal jitter at strengths 0.01, 0.03, 0.05 and 0.1.

## Spikformer v1

Use `lif`, `block` (ParaMuT), or `psn` with the same backbone and training recipe. These runs use full-sequence batch normalization, AdamW, warmup, cosine decay, mixup, and FP32. CIFAR10 uses T4, seed 42, batch 128, embedding 384, 4 transformer blocks, 12 heads and learning rate 0.0005. CIFAR10-DVS uses T16, batch 16, embedding 256, 2 blocks, 16 heads and learning rate 0.001. The DVS split reserves the final 10% of each class in dataset order. Inputs preserve event counts at 128 × 128. The LIF control uses input decay and a detached hard reset. ParaMuT uses a fixed unit diagonal with residual cap 0.25, and defaults to K2 for CIFAR10 or K8 for CIFAR10-DVS. [Upstream architecture and recipes](https://github.com/ZK-Zhou/spikformer/tree/621268981827af47aa666c1b5f774de4281d90ad)

```bash
python run_spikformer_reproduction.py --dataset cifar10 --method lif \
  --data-dir data/cifar10 --output-dir outputs/spikformer/cifar10/lif --download

python run_spikformer_reproduction.py --dataset cifar10dvs --method block \
  --data-dir data/cifar10dvs --output-dir outputs/spikformer/cifar10dvs/k8-seed2021 \
  --seed 2021 --block-size 8 --epochs 96 --cooldown-epochs 104

python run_spikformer_robustness.py \
  --checkpoint outputs/spikformer/cifar10dvs/k8-seed2021/best.pt \
  --data-dir data/cifar10dvs --output-dir outputs/spikformer/cifar10dvs/k8-seed2021/robustness
```

CIFAR10 defaults to 400 cosine epochs and 10 cooldown epochs. For the 200-epoch CIFAR10-DVS table, run the illustrated 96 + 104 schedule for seeds 2021, 2022 and 2023, including a separate `--method block --block-size 4` run. The original 2021 LIF, K8 and PSN-style runs extended 106-epoch checkpoints while preserving optimizer and random state. K4 and all 2022 and 2023 runs trained from scratch. The shorter DVS protocol uses the default 96 + 10 schedule. Both protocols select the best test accuracy. Keep `best.pt` and `summary.json` together for robustness evaluation.

The Spikformer evaluator scores FGSM and 20-step PGD with two starts, ε/4 steps, and native and full-state surrogate gradients. The union accuracy counts an example as broken if any tested candidate succeeds. CIFAR budgets are in raw pixel units. DVS budgets are in fractional raw event-count units up to 1. Published curves and AUCs include the zero-budget endpoint. The output directory contains `robustness.csv` (percent), predictions, and a summary with normalized AUCs.
