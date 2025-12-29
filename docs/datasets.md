# Datasets

DEMIS supports two types of input datasets: the synthetic **EM424 dataset** and generic EM image grids.

## EM424 Dataset

The EM424 dataset is a synthetic dataset created specifically for training and evaluating the DEMIS tool. It consists
of 424 distinct high-quality and high-resolution EM images publicly available on [EMPIAR](https://www.ebi.ac.uk/empiar/)
or [The Cell Image Library](http://www.cellimagelibrary.org/).

### Structure

Each selected image was divided into a grid of overlapping image tiles of size 1024×1024 pixels. Additionally, random
brightness and contrast changes, random rotation, random translation, and Gaussian noise were applied to each tile.

The expected directory structure is:

- `images/`: Individual image tiles.
- `indices/`: Indices for training LoFTR.
- `labels/`: Ground-truth labels containing tile poses and grid metadata.
- `splits/`: Lists of grids in each split (train/val/test).

### Generation

You can generate a new version of the EM424 dataset using the provided scripts:

1.  [synthesize_em424.py](../scripts/synthesize_em424.py):
    Synthesizes EM424 from a directory containing source EM images.

    ```bash
    uv run scripts/synthesize_em424.py <source_images_dir> <output_dir>
    ```

2.  [generate_em424_splits.py](../scripts/generate_em424_splits.py):
    Generates split metadata and indices for training.
    ```bash
    uv run scripts/generate_em424_splits.py
    ```

References of source images used for generating EM424 can be found in [em424-references.md](em424-references.md).

## Generic Datasets

Other datasets containing unlabeled grids of overlapping EM images are also supported.

### Dataset Structure

- **Directory**: The expected grid size can be either given via parameters or derived from the name of the dataset
directory, if formatted as `<rows>x<cols>` (e.g., `2x2`).
- **Files**: Filenames of image tiles should be formatted as
`<dataset_name>_g<grid_index>_t<tile_index>_s<slice_index>.tif`.

### Conversion

A utility script is available to convert results from other tools (currently only
[MIST](https://github.com/usnistgov/MIST)) to the EM424 format for comparison:

```bash
uv run scripts/results_converter.py <tool_type> <input_dir> <output_dir>
```
