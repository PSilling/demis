# DEMIS: Deep Electron Microscopy Image Stitching

Electron microscopy (EM) image stitching based on [LoFTR](https://github.com/zju3dv/LoFTR)
feature matching. Details presented on [Excel@FIT 2023](https://excel.fit.vutbr.cz/)
([Poster](https://excel.fit.vutbr.cz/submissions/2023/015/15_poster.pdf),
[Commentary](https://excel.fit.vutbr.cz/submissions/2023/015/15.pdf)), a student
conference held by the Faculty of Information Technology, Brno University of Technology.

## Environment Setup

The following commands can be used to prepare the working environment. A CUDA-enabled
machine is necessary.

```
conda env create -f environment.yml
conda activate demis
```

Moreover, weights for LoFTR need to be placed in `LoFTR/weights/`. The following
weights are recommended:

  - [demis_ds.ckpt](https://drive.google.com/file/d/1HYnYKTxnAA5g7Tizk0Ney_Jeq_VoJYUu/view?usp=sharing)
    &ndash; Weights fine-tuned on the DEMIS dataset (described below).
  - [outdoor_ds.ckpt](https://drive.google.com/file/d/1M-VD35-qdB5Iw-AtbDBCKC7hPolFW9UY/view?usp=sharing)
    &ndash; Pre-trained outdoor dual-softmax weights provided by LoFTR. Trained
    on conventional photography.

## Method Overview

The DEMIS tool stitches images in the following way.

  1. The brightness and contrast of raw image tiles from a grid of overlapping
     EM images are normalised.
  
  2. Feature matches are detected between pairs of adjacent tiles by LoFTR.
  
  3. Pairwise transformations are estimated from the detected feature matches.
  
  4. The pairwise transformations are optimised globally by
     [graphslam](https://github.com/JeffLIrion/python-graphslam/).
  
  5. The tiles in the grid are stitched together using the optimised transformations.

Stitching using a minimum-spanning tree (MST) instead of SLAM optimisation is also
supported. The methods and their parameters (such as resolution scaling, expected tile
overlaps, and types of estimated transformations) can be adjusted by YAML configuration
files (examples can be found in `configs/`). Paths to data can be configured in the
same way.

## Input Datasets

The tool supports two types of input datasets.
  
  1. **DEMIS dataset** &ndash; A synthetic dataset created specifically for training and
  evaluating the DEMIS tool. The DEMIS dataset can be stitched using ground-truth labels
  if desired.
  
  2. **Other datasets** &ndash; Standard datasets containing grids of overlapping EM images.
  The expected grid size is derived from the name of the containing directory, which
  should be formatted as `<rows>x<cols>`. Filenames of image tiles should be formatted
  as `<dataset_name>_g<grid_index>_t<tile_index>_s<slice_index>.tif`.

By default, input datasets should be placed in `datasets/`.

## DEMIS Dataset

A synthetic dataset created by manually selecting 424 distinct high-quality and
high-resolution EM images publicly available on [EMPIAR](https://www.ebi.ac.uk/empiar/)
or [The Cell Image Library](http://www.cellimagelibrary.org/). Each selected image was
divided into a grid of overlapping image tiles of size 1024&times;1024 pixels. Additionally,
random brightness and contrast changes, random rotation, random translation, and Gaussian
noise were applied to each tile. The dataset and its source images can be downloaded
from FIT NextCloud: [source images](https://nextcloud.fit.vutbr.cz/s/773XXbQdYBGKxKH),
[DEMIS dataset](https://nextcloud.fit.vutbr.cz/s/R3GAr2JcSQFeyi6). The references for
the source images images can be found in [docs/demis-references.md](docs/demis-references.md).

It is also possible to generate a new version of the DEMIS dataset using the following
scripts.

  1. [scripts/synthesize_demis.py](scripts/synthesize_demis.py) &ndash; Synthesises DEMIS
  from the directory that contains the EM images to split.
  
  2. [scripts/generate_demis_splits.py](scripts/generate_demis_splits.py) &ndash; Generates
  split metadata of the DEMIS dataset, including indices needed for training LoFTR.

The scripts should be executed as modules from the root directory. For example:

```
python3 -m scripts.synthesize_demis <directory_with_source_images> <output_directory>
python3 -m scripts.generate_demis_splits configs/demis-fine-tuned.yaml
```

The expected directory structure is the following.

  - `images/` &ndash; Individual image tiles.
  - `indices/` &ndash; Indices for training LoFTR.
  - `labels/` &ndash; Ground-truth labels containing tile poses and grid metadata.
  - `splits/` &ndash; Lists of grids in each split.

## Usage

The source codes are available in `src/`. The main [stitch.py](scripts/stitch.py) script
for image stitching is located in `scripts/` and should be executed as a module from
the root directory. For example, the following command starts stitching the DEMIS dataset
using default settings and the fine-tuned LoFTR model. By default, the stitched images
will be saved to `output/DEMIS/`.

```
conda activate demis
python3 -m scripts.stitch configs/demis-fine-tuned.yaml
```

Additionally, a Jupyter notebook is provided for ease of use:
[notebooks/stitch.ipynb](notebooks/stitch.ipynb).

## Evaluation

Evaluation on the DEMIS dataset can be started using the following commands.
The evaluation script compares the DEMIS tool to a SIFT baseline. Apart from the
feature matching method, both evaluated solutions rely on the same DEMIS tool
configuration.

Fine-tuned weights:

```
conda activate demis
python3 -m scripts.evaluate_demis configs/eval-demis-fine-tuned.yaml
```

Pre-trained weights:

```
conda activate demis
python3 -m scripts.evaluate_demis configs/eval-demis-pre-trained.yaml
```

## Fine-Tuning LoFTR

The LoFTR module was fine-tuned on training data from the DEMIS dataset. To start
the fine-tuning process on 10 epochs and an initial learning rate of `1e-5`, the
following commands can be used. The default configuration expects a single machine
with two GPUs. Logs and checkpoints will be saved to `LoFTR/logs/`.

```
conda activate demis
cd LoFTR/
bash scripts/reproduce_train/demis.sh
```

To enable training on the DEMIS dataset, the following DEMIS-specific files were
added to the official implementation of LoFTR.
  
  - [LoFTR/scripts/reproduce_train/demis.sh](LoFTR/scripts/reproduce_train/demis.sh)
    &ndash; DEMIS training execution script.

  - [LoFTR/configs/loftr/demis/loftr_demis_dense.py](LoFTR/configs/loftr/demis/loftr_demis_dense.py)
    &ndash; DEMIS training configuration.
  
  - [LoFTR/configs/data/demis_trainval.py](LoFTR/configs/data/demis_trainval.py)
    &ndash; DEMIS dataset structure specification.
  
  - [LoFTR/src/datasets/demis.py](LoFTR/src/datasets/demis.py)
    &ndash; DEMIS dataset loader class.

More details can be found in the official
[training documentation](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md)
of LoFTR. 

## References

Sun, J., Shen, Z., Wang, Y., Bao, H. and Zhou, X. LoFTR: Detector-Free Local
Feature Matching with Transformers. In: Conference on Computer Vision and
Pattern Recognition. Nashville, TN, USA: IEEE, June 2021, p. 8918&ndash;8927.
CVPR, no. 2021. DOI: 10.1109/CVPR46437.2021.00881. ISSN 1063-6919.
