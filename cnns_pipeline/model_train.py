import sys
import argparse
import numpy as np
import tensorflow as tf

from pathlib import Path
from typing import Tuple, Dict, List
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

try:
    sys.path.insert(1, str(Path(__file__).parent.parent))
except Exception:
    raise EnvironmentError

from utils.paths_manager import PathsManager
from cells_segmentation.losses import iou, dice, JaccardLoss
from cells_segmentation.models.fully_convolutional_network import FCN
from cells_segmentation.models.shallow_unet import ShallowUnet
from cells_segmentation.models.attention_dual_path import AttentionDualPathUnet
from utils.constants import SEED


def configure_dataset(
        dataset: tf.data.Dataset,
        shuffle: bool,
        buffer_size: int,
        batch_size: int
) -> tf.data.Dataset:
    """
    Configures the dataset processing pipeline for its performance enhancement.
    """
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=SEED)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def parse_image(
        path: Path,
        channels: int,
        mask: bool = False
) -> tf.Tensor:
    """
    Converts image into tf.Tensor.
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=channels)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if mask:
        image = tf.where(
            image > 0.0,
            np.dtype('float32').type(1),
            np.dtype('float32').type(0)
        )
    return image


def process_images(
        image_path: Path
) -> tuple:
    """
    Processes images fetched  by the tf.Dataset instance.
    """
    pos_click_map_path = tf.strings.regex_replace(image_path, "image", "pos_click")
    neg_click_map_path = tf.strings.regex_replace(image_path, "image", "neg_click")
    mask_path = tf.strings.regex_replace(image_path, "image", "mask")

    # Parse images
    hpa_image = parse_image(path=image_path, channels=3) / 255.0
    pos_click_map = parse_image(path=pos_click_map_path, channels=1)
    neg_click_map = parse_image(path=neg_click_map_path, channels=1)
    seg_mask = parse_image(path=mask_path, channels=1, mask=True)

    input_tensor = tf.concat([hpa_image, pos_click_map, neg_click_map], axis=2)

    return input_tensor, seg_mask


def create_training_datasets(
        data_path: Path,
        batch_size: int,
        buffer_size: int,
        split_ratio: float,
        seed: int
) -> Tuple[Dict[str, tf.data.Dataset], int, int]:
    """
    Creates train and validation datasets.
    """
    assert 0.0 < split_ratio < 1.0

    autotune = tf.data.experimental.AUTOTUNE
    full_dataset = tf.data.Dataset.list_files(str(data_path / "image/*.png"), seed=seed)
    full_dataset = full_dataset.map(process_images, num_parallel_calls=autotune)

    dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
    train_dataset_size = int(split_ratio * dataset_size)
    val_dataset_size = dataset_size - train_dataset_size

    train_dataset = full_dataset.take(train_dataset_size)
    remaining = full_dataset.skip(train_dataset_size)
    val_dataset = remaining.take(val_dataset_size)

    dataset = {"train": train_dataset, "val": val_dataset}

    dataset['train'] = configure_dataset(
        dataset=dataset['train'], shuffle=True, buffer_size=buffer_size, batch_size=batch_size
    )
    dataset['val'] = configure_dataset(
        dataset=dataset['val'], shuffle=False, buffer_size=buffer_size, batch_size=batch_size
    )

    return dataset, train_dataset_size, val_dataset_size


def make_callbacks(
        model_name: str,
        models_dir: Path,
        es_patience: int
) -> List[tf.keras.callbacks.Callback]:
    """
    Makes list of callbacks for model training.
    """
    save_path = str(models_dir / f'{model_name}.hdf5')

    return [
        EarlyStopping(
            monitor="val_loss", patience=es_patience, mode="min", verbose=1),
        ModelCheckpoint(
            filepath=save_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
    ]


if __name__ == '__main__':
    paths_manager = PathsManager()
    config = paths_manager.config
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str, default=config.model, choices=['FCN', 'UNET', 'DP-UNET'],
        help='')
    parser.add_argument(
        '--epochs', type=int, default=config.epochs,
        help='')
    parser.add_argument(
        '--batch_size', type=int, default=config.batch_size,
        help='')
    parser.add_argument(
        '--buffer_size', type=int, default=config.buffer_size,
        help='')
    parser.add_argument(
        '--split_ratio', type=float, default=config.split_ratio,
        help='')
    parser.add_argument(
        '--es_patience', type=float, default=config.es_patience,
        help='')
    parser.add_argument(
        '--seed', type=int, default=config.seed,
        help='')
    args = parser.parse_args()

    # Create training and validation datasets
    seg_dataset, train_size, val_size = create_training_datasets(
        data_path=paths_manager.train_dataset_dir(),
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        split_ratio=args.split_ratio,
        seed=args.seed
    )

    # Build the segmentation model
    if args.model == 'FCN':
        model = FCN()
    elif args.model == 'UNET':
        model = ShallowUnet()
    elif args.model == 'DP-UNET':
        model = AttentionDualPathUnet()
    else:
        raise NotImplementedError()

    model.compile(
        loss_function=JaccardLoss(),
        optimizer=tf.optimizers.Adam(),
        metrics=[iou, dice]
    )

    callbacks_list = make_callbacks(
        model_name=model.__class__.__name__,
        models_dir=paths_manager.models_dir(),
        es_patience=args.es_patience
    )

    # Train the model on GPU
    with tf.device("device:GPU:0"):
        model.train(
            dataset=seg_dataset,
            train_size=train_size,
            val_size=val_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=callbacks_list
        )
