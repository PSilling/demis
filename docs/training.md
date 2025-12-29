# Training LoFTR

The LoFTR module in DEMIS can be fine-tuned on the EM424 dataset to improve feature matching performance on electron microscopy images.

## Fine-Tuning Process

To start the fine-tuning process, use the provided [reproduction script](../LoFTR/scripts/reproduce_train/demis.sh).
The default configuration expects a single machine with two GPUs.

```bash
cd LoFTR/
bash scripts/reproduce_train/demis.sh
```

Logs and checkpoints will be saved to `LoFTR/logs/`.

## Configuration Files

The training process is controlled by several configuration files added to the official LoFTR implementation:

- [LoFTR/scripts/reproduce_train/demis.sh](../LoFTR/scripts/reproduce_train/demis.sh): Training execution script.
- [LoFTR/configs/loftr/demis/loftr_demis_dense.py](../LoFTR/configs/loftr/demis/loftr_demis_dense.py): Training configuration.
- [LoFTR/configs/data/demis_trainval.py](../LoFTR/configs/data/demis_trainval.py): Dataset structure specification.
- [LoFTR/src/datasets/demis.py](../LoFTR/src/datasets/demis.py): EM424 dataset loader.

For more details on the training pipeline, refer to the official [LoFTR training documentation](../LoFTR/docs/TRAINING.md).
