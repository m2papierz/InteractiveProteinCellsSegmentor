import sys
import argparse
import tensorflow as tf

from glob import glob
from pathlib import Path
from typing import Tuple, Dict

try:
    sys.path.insert(1, str(Path(__file__).parent.parent))
except Exception:
    raise EnvironmentError

from utils.paths_manager import PathsManager
from cells_segmentation.losses import iou, dice, JaccardLoss
from cells_segmentation.models.shallow_unet import ShallowUnet
from cells_segmentation.models.fully_convolutional_network import FCN
from cells_segmentation.models.attention_dual_path import AttentionDualPathUnet

from cnns_pipeline.model_train import process_images
from cnns_pipeline.model_train import configure_dataset
from utils.constants import SEED


def create_test_dataset(
        data_path: Path,
        buffer_size: int
) -> Tuple[Dict[str, tf.data.Dataset], int]:
    """
    Creates test dataset.
    """
    autotune = tf.data.experimental.AUTOTUNE
    dataset_size = len(glob(str(data_path / "image/*.png")))

    test_dataset = tf.data.Dataset.list_files(str(data_path / "image/*.png"), seed=SEED)
    test_dataset = test_dataset.map(process_images, num_parallel_calls=autotune)

    dataset = {"test": test_dataset}
    dataset['test'] = configure_dataset(
        dataset=dataset['test'], shuffle=False, buffer_size=buffer_size, batch_size=1)

    return dataset, dataset_size


def evaluate_model(
        model: tf.keras.models.Model,
        dataset: Dict[str, tf.data.Dataset],
        data_size: int,
        batch: int
) -> None:
    """
    Evaluates the model.
    """
    test_steps = tf.floor(data_size / batch)
    loss, iou_score, dice_score = model.evaluate(dataset['test'], steps=test_steps)

    print(f"Loss: {loss}")
    print(f"IOU: {iou_score}")
    print(f"Dice: {dice_score}")


if __name__ == '__main__':
    paths_manager = PathsManager()
    config = paths_manager.config
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str, default=config.model, choices=['FCN', 'UNET', 'DP-UNET'],
        help='')
    parser.add_argument(
        '--batch_size', type=int, default=config.batch_size,
        help='')
    parser.add_argument(
        '--buffer_size', type=int, default=config.buffer_size,
        help='')
    args = parser.parse_args()

    seg_dataset, test_size = create_test_dataset(
        data_path=paths_manager.test_dataset_dir(),
        buffer_size=args.buffer_size
    )

    custom_objects = {
        JaccardLoss.__name__: JaccardLoss(),
        iou.__name__: iou,
        dice.__name__: dice
    }

    if args.model == 'UNET':
        model_path = paths_manager.models_dir() / f'{ShallowUnet.__name__}.hdf5'
    elif args.model == 'FCN':
        model_path = paths_manager.models_dir() / f'{FCN.__name__}.hdf5'
    elif args.model == 'DP-UNET':
        model_path = paths_manager.models_dir() / f'{AttentionDualPathUnet.__name__}.hdf5'
    else:
        raise NotImplementedError()

    best_model = tf.keras.models.load_model(
        model_path, custom_objects=custom_objects)
    best_model.compile(
        loss=JaccardLoss(),
        optimizer=tf.optimizers.Adam(),
        metrics=[iou, dice]
    )

    evaluate_model(
        model=best_model,
        dataset=seg_dataset,
        data_size=test_size,
        batch=args.batch_size
    )
