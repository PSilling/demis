"""Evaluates the DEMIS pipeline on the DEMIS dataset."""
from src.config.config import get_cfg_defaults
from src.eval.demis_evaluator import DEMISEvaluator


# TODO: Add CLI selection of evaluation methods.
if __name__ == "__main__":
    demis_config_path = "configs/eval.yaml"
    eval_matching = True
    eval_homography = True
    eval_grid = True
    count = None
    cfg = get_cfg_defaults()
    cfg.merge_from_file(demis_config_path)

    evaluator = DEMISEvaluator(cfg, eval_matching, eval_homography,
                               eval_grid, count)
    evaluator.evaluate()
