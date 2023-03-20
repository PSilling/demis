import cv2


class ImageCache:
    """Cache for image tiles.
    TODO: Actual caching.
    """

    def __init__(self, cfg):
        """ImageCache constructor.

        :param cfg: DEMIS configuration.
        """
        self.cfg = cfg

        # Setup CLAHE normalisation.
        # TODO: More automated intensity normalisation.
        if cfg.STITCHER.NORMALISE_INTENSITY:
            self.clahe = cv2.createCLAHE(
                clipLimit=cfg.STITCHER.CLAHE_LIMIT,
                tileGridSize=cfg.STITCHER.CLAHE_GRID
            )

    def load_img(self, path, mode=cv2.IMREAD_GRAYSCALE):
        """Loads an image at the given path. Configured image processing, such as
        intensity normalisation, is applied automatically upon image loading.

        :param path: Path to the image.
        :param mode: Read mode.
        :return: Image matrix.
        """
        img = cv2.imread(path, mode)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")

        # Setup CLAHE normalisation.
        if self.cfg.STITCHER.NORMALISE_INTENSITY:
            img = self.clahe.apply(img)

        return img
