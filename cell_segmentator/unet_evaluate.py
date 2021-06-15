import tensorflow as tf

from unet_train import create_dataset
from utils import show_predictions
from utils import combined_loss, iou_loss, dice_loss

# Paths
DATA_PATH = "D:/DataScience/THESIS/Data/HPA_segmentation/prepared/"
BEST_MODEL_PATH = "D:/DataScience/THESIS/models/20210614-235209best_segmentation_model.hdf5"

# General
BATCH_SIZE = 8


def evaluate_model(model: tf.keras.models.Model, dataset: dict, test_size: int, batch_size: int) -> None:
    test_steps = tf.floor(test_size / batch_size)
    combined, iou, dice = model.evaluate(dataset['test'], steps=test_steps)

    print(f"Combined_loss: {combined}")
    print(f"IOU loss: {iou}")
    print(f"Dice loss: {dice}")


def main():
    segmentation_dataset, train_size, val_size = create_dataset(DATA_PATH)
    samples = segmentation_dataset['test'].take(1)
    custom_objects = {combined_loss.__name__: combined_loss,
                      iou_loss.__name__: iou_loss,
                      dice_loss.__name__: dice_loss}

    best_model = tf.keras.models.load_model(BEST_MODEL_PATH, custom_objects=custom_objects)
    evaluate_model(model=best_model,
                   dataset=segmentation_dataset,
                   test_size=val_size,
                   batch_size=BATCH_SIZE)

    show_predictions(model=best_model, sample_images=samples)


if __name__ == '__main__':
    main()
