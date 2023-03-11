"""Synthesizes the DEMIS dataset."""
import re
from src.dataset.demis_synthesizer import DEMISSynthesizer, DEMISSynthesizerConfig


# TODO: Replace with CLI parameterization.
if __name__ == "__main__":
    config = DEMISSynthesizerConfig()
    tile_resolution = "1024x1024"

    # Parse the expected tile resolution.
    match = re.match(r"(\d+)x(\d+)", tile_resolution, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Invalid resolution specification: {tile_resolution}")
    config.TILE_RESOLUTION = tuple(int(x) for x in match.groups())

    synthesizer = DEMISSynthesizer(config)
    synthesizer.synthesize_demis()
