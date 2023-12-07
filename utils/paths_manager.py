import os

from pathlib import Path
from omegaconf import OmegaConf


class PathsManager:
    def __init__(self):
        try:
            self.config = OmegaConf.load(Path(__file__).parent.parent / "config.yaml")
        except Exception as e:
            print(f"Error loading the configuration file: {e}")
            print("Please create a 'config.yaml' file with the required configuration.")

        self._root_path = Path(__file__).parent.parent

    @staticmethod
    def _ensure_directory(directory_path: Path) -> Path:
        if not directory_path.exists():
            os.makedirs(directory_path)
        return directory_path

    def _datasets_dir(self) -> Path:
        path = self._root_path / self.config.data_dir / 'datasets'
        return self._ensure_directory(path)

    def train_dataset_dir(self) -> Path:
        path = self._datasets_dir() / 'train_valid'
        return self._ensure_directory(path)

    def train_pos_click_maps_dir(self) -> Path:
        path = self.train_dataset_dir() / 'pos_click'
        return self._ensure_directory(path)

    def train_neg_click_maps_dir(self) -> Path:
        path = self.train_dataset_dir() / 'neg_click'
        return self._ensure_directory(path)

    def test_dataset_dir(self) -> Path:
        path = self._datasets_dir() / 'test'
        return self._ensure_directory(path)

    def test_pos_click_maps_dir(self) -> Path:
        path = self.test_dataset_dir() / 'pos_click'
        return self._ensure_directory(path)

    def test_neg_click_maps_dir(self) -> Path:
        path = self.test_dataset_dir() / 'neg_click'
        return self._ensure_directory(path)

    def train_annotations_dir(self) -> Path:
        path = self._root_path / self.config.data_dir / 'train_annotations'
        return self._ensure_directory(path)

    def test_annotations_dir(self) -> Path:
        path = self._root_path / self.config.data_dir / 'test_annotations'
        return self._ensure_directory(path)

    def models_dir(self) -> Path:
        path = self._root_path / self.config.models_dir
        return self._ensure_directory(path)

    def test_img_path(self) -> Path:
        path = self._root_path / 'utils/test_image/test_img.png'
        return Path(path)

    def cursor_imgs_path(self) -> Path:
        path = self._root_path / 'utils/pyqt_cursor_imgs'
        return Path(path)
