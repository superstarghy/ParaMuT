# SHD and UCI HAR

Run the commands from the repository root. `run_sequence.py train` writes `best.pt`, the per-epoch `history.jsonl` and `training.json` to a new directory. `robustness` loads the model specification from that checkpoint and writes accuracies as fractions. Both commands select CUDA when available and accept `--device cpu`.

## Spiking Heidelberg Digits

Prepare the official train and test files (8,156 and 2,264 spoken digits). The preparation command checks each HDF5 file against its SHA-256 checksum and produces a frame cache. If you already have `shd_train.h5` and `shd_test.h5`, place them in `data/shd/raw` and omit `--download`. [SHD dataset and citation](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)

```bash
python scripts/prepare_shd.py --source data/shd/raw --output data/shd/frames --download
```

### DCLS and zero-delay networks

The model uses 10 ms frames with 140 input channels. Each frame sums five adjacent cochlear channels. The three synapses map 140 → 256 → 256 → 20 and each appends a 12-step tail. Inputs retain nonnegative counts. The test set is ordered by class and original sample index. Evaluation microbatches retain the padded length of their 256-sample parent batch because padding affects the classifier. [SNN-delays backbone](https://github.com/Thvnvtos/SNN-delays/tree/d169b4e3049a3d5bff56c84a8b2f0c4e835aafda) · [SpikingJelly framing reference](https://github.com/fangwei123456/spikingjelly/blob/01823bed8d14c3149a89cae310ae9e08f0dccb34/spikingjelly/datasets/shd.py)

```bash
python run_sequence.py train --dataset shd --variant dcls_pm_k16_cap075 --seed 0 \
    --data data/shd/frames --output runs/shd-pm16-s0
python run_sequence.py robustness --checkpoint runs/shd-pm16-s0/best.pt \
    --data data/shd/frames --output runs/shd-pm16-s0/robustness.json
```

| Synapses | LIF | ParaMuT |
| --- | --- | --- |
| Learned delays | `dcls_lif` | `dcls_pm_k{K}_{constraint}` |
| Zero delay | `lif` | `pm_k{K}_{constraint}` |

Replace `{K}` with `4`, `16` or `32`. The constraint is `none`, `cap075` (maximum row and column L1 norms of 0.75) or `db` (fixed diagonal 0.5 and strict-lower residual capped at 0.25).

The study used 150 epochs, batch size 256 and seeds 0, 1 and 2. Checkpoint selection uses the unweighted mean of test-batch accuracies. Reported clean accuracy weights each sample equally. The full-test FGSM/PGD7 grid in raw-count units is 0, .005, .01, .02, .025, .05, .075, .1, .125, .15, .175 and .2. Evaluations use integer delay kernels and preserve zero-valued artificial padding.

### PSN-style on SpArch

The archived SHD PSN-style experiment uses a different backbone. It bins the original 700 channels into 100 fixed time steps over 1.4 s. Two 128-unit layers mix entire sequences using noncausal matrices and feed a leaky readout with random initial state. Its default budget is 100 epochs, batch size 128 for training and 32 for robustness. Selection uses the mean of the test-batch accuracies. Fresh clean accuracy evaluates the selected checkpoint with a new draw of the readout state. [SpArch source](https://github.com/idiap/sparch/tree/f8756254ab8d0e3337eb69542c684b922d6b6cbd) · [SpArch license](../../licenses/SpArch-BSD-3-Clause.txt)

```bash
python run_sequence.py train --dataset shd --variant psn --seed 0 \
    --data data/shd/raw --output runs/shd-psn-s0
python run_sequence.py robustness --checkpoint runs/shd-psn-s0/best.pt \
    --data data/shd/raw --output runs/shd-psn-s0/robustness.json
```

Seeds 0, 1 and 2 use the official train and test splits. Attacks use four fixed readout-state draws to optimize each perturbation and eight independent paired draws to evaluate it. The full-test grid is 0, .0001, .00025, .0005, .001, .0025, .005 and .01 in raw-count units. The resulting AUC covers a different range from the DCLS study.

## UCI Human Activity Recognition

The preparation command downloads the UCI archive, preserves the 7,352 training and 2,947 test windows from disjoint subjects, and fits channel means and standard deviations on the training windows. Each sample has 128 observations of nine inertial channels at 50 Hz. An existing official archive can be placed at `data/uci-har/official.zip`. [UCI dataset and citation](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

```bash
python scripts/prepare_uci_har.py --root data/uci-har
python run_sequence.py train --dataset uci-har --variant k16 --seed 0 \
    --data data/uci-har --output runs/har-pm16-s0
python run_sequence.py robustness --checkpoint runs/har-pm16-s0/best.pt \
    --data data/uci-har --output runs/har-pm16-s0/robustness.json
```

SpikeCNN-BN has three causal convolutional stages (32, 64 and 64 channels), each with batch normalization, a spiking neuron and max pooling. A mean over time feeds the six-class linear readout. The study used Adam at 0.001, a cosine schedule, 120 epochs, batch size 128 and seed 0. The checkpoint with the best sample-weighted test accuracy supplies the reported metrics. [SpikeCNN architecture reference](https://doi.org/10.3389/fnins.2023.1233037)

| Model | Accepted identifiers |
| --- | --- |
| Sequential LIF | `lif` |
| PSN-style, full sequence | `psn` |
| ParaMuT | `k{K}` |
| ParaMuT with a matrix cap | `k{K}_c0p5`, `k{K}_c0p75`, `k{K}_c0p9` |
| ParaMuT with fixed diagonal 0.5 and residual cap 0.25 | `k{K}_db` |

For HAR, `{K}` is `4`, `8`, `16` or `32`. The full-test FGSM/PGD7 grid is 0, .01, .025, .05, .1 and .2 in standard deviations per channel. Inputs are signed. PGD takes seven steps of size ε/4 with one deterministic random start. For both HAR and DCLS, the evaluator also considers the clean input, FGSM and every PGD iterate when selecting the strongest candidate per example.

Normalized AUC is the trapezoidal area under accuracy as a function of epsilon, divided by the width of that grid. All three protocols use the official test split for both checkpoint selection and reporting. [Recorded results and curves](../../results/README.md)
