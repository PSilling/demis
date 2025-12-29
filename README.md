# DEMIS: Deep Electron Microscopy Image Stitching

DEMIS is a tool for stitching grids of **electron microscopy** (EM) images using deep learning-based feature matching
and optical flow refinement.

<a href="https://tescan.com/" target="_blank">
  <img src="./assets/logo-tescan.png" alt="TESCAN GROUP logo" width="240" />
</a>
&nbsp;
<a href="https://www.fit.vut.cz/.en" target="_blank">
  <img src="./assets/logo-fit-but.png" alt="FIT BUT logo" width="240" />
</a>

## Overview

DEMIS normalizes image tiles, detects feature matches using [LoFTR](https://github.com/zju3dv/LoFTR), estimates pairwise
transformations between tiles, and optimizes them globally to stitch the final grid.

Key features:

- **Deep feature matching**: Uses LoFTR for robust matching of EM images with low-quality or highly repetitive texture.
- **Optical flow refinement**: Refines alignments using [RAFT](https://github.com/princeton-vl/RAFT)-based optical flow.
- **Flexible configuration**: Support for multiple feature matching, grid construction and image compositing methods.
- **ImageJ integration**: Includes a plugin for easy use within [ImageJ2](https://imagej.net/software/imagej2/) and
[Fiji](https://imagej.net/software/fiji/).

## Installation

The following setup guide was tested on **Ubuntu 22.04** and **Windows 11**. Running on Windows might have a negative
performance impact.

### Prerequisites

- **Python 3.9**
- **uv**: DEMIS uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management.
- **CUDA**: A CUDA-enabled GPU is required to use deep learning-based features. Running on CPU would be too slow.

### Setup

1. Clone the repository:

```bash
git clone https://github.com/PSilling/demis.git
cd demis
```

2. Install dependencies using `uv`:

```bash
uv sync
```

3. Set `PYTHONPATH` to the root of the repository.

- **Linux**: `export PYTHONPATH=$PWD`
- **Windows**: `$env:PYTHONPATH = $PWD` (PowerShell)

4. Download weights for LoFTR:

- Place LoFTR weights in `LoFTR/weights/`. Recommended options:

  - [demis_ds.ckpt](https://drive.google.com/file/d/1HYnYKTxnAA5g7Tizk0Ney_Jeq_VoJYUu/view?usp=sharing) &ndash; weights
  fine-tuned on the [EM424 dataset](docs/datasets.md) (synthetic EM image grids).
  - [outdoor_ds.ckpt](https://drive.google.com/file/d/1M-VD35-qdB5Iw-AtbDBCKC7hPolFW9UY/view?usp=sharing) &ndash;
  original outdoor dual-softmax weights of LoFTR. Trained on conventional photography only.

## Usage

Scripts from `scripts/` directory can be run using `uv run`. Ensure `PYTHONPATH` is set to project root.

### Stitching

To stitch a dataset using a configuration file:

```bash
uv run scripts/stitch.py --config configs/em424-fine-tuned.yaml
```

### Evaluation

To evaluate performance on the EM424 dataset:

```bash
uv run scripts/evaluate_demis.py --config configs/em424-fine-tuned.yaml
```

## Documentation

Detailed documentation is available in the `docs/` directory:

- [**Configuration Guide**](docs/configuration.md): Detailed explanation of all YAML options.
- [**Datasets**](docs/datasets.md): Information on the EM424 dataset and expected dataset format.
- [**Training**](docs/training.md): Instructions for fine-tuning LoFTR on EM424 dataset.
- [**ImageJ Plugin**](imagej/README.md): Guide for building and using the ImageJ plugin.

## Acknowledgements

DEMIS is supported by [TESCAN GROUP](https://tescan.com/) and the [Faculty of Information Technology, Brno University of Technology](https://www.fit.vut.cz/.en).

## Citation

If you use our tool, please cite our [paper](https://www.scitepress.org/Papers/2025/133149/133149.pdf) published in
**BIOIMAGING 2025**:

ŠILLING, P.; ŠPANĚL, M. DEMIS: Electron Microscopy Image Stitching using Deep Learning Features and Global Optimisation.
Proceedings of the 18th International Joint Conference on Biomedical Engineering Systems and Technologies - BIOIMAGING.
Porto: Institute for Systems and Technologies of Information, Control and Communication, 2025. p. 255-256.
ISBN: 978-989-758-731-3.
