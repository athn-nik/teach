import logging
import hydra
from omegaconf import DictConfig
import teach.launch.prepare  # noqa

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="canonicalize_motion")
def _canonicalize_motion(cfg: DictConfig):
    return canonicalize_motion(cfg)


def canonicalize_motion(cfg: DictConfig) -> None:
    from pathlib import Path
    infolder = Path(cfg.infolder)
    outfolder = Path(cfg.outfolder)
    outfolder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Canonicalize motion from {infolder} to {outfolder}")

    logger.info("Loading libraries")
    # lazy loading
    import os
    from tqdm import tqdm
    from teach.data.pose2joints import Rifke
    import numpy as np
    import torch
    logger.info("Libraries loaded")

    rifke = Rifke(normalization=False)
    for infile in tqdm(infolder.glob("*.npy"), desc="Processing"):
        # save folder
        outfile = outfolder / os.path.basename(infile)

        # loading data
        data = torch.from_numpy(np.load(infile))

        # canonicalize the data
        # => removing the first rotation
        candata = rifke(rifke.inverse(data)).numpy()
        np.save(outfile, candata)

    logger.info("All the canonicalization are done.")


if __name__ == '__main__':
    _canonicalize_motion()
