import tensorflow as tf

from unet_train import create_dataset
from utils import show_predictions
from utils import combined_iou_dice_loss, iou, dice

# Paths
DATA_PATH = "D:/DataScience/THESIS/Data/HPA_segmentation/prepared/"
BEST_MODEL_PATH = "D:/DataScience/THESIS/models/best_segmentation_model.hdf5"


def main():
    segmentation_dataset, train_size, val_size = create_dataset(DATA_PATH)
    samples = segmentation_dataset['train'].take(1)
    best_model = tf.keras.models.load_model(BEST_MODEL_PATH,
                                            custom_objects={combined_iou_dice_loss.__name__: combined_iou_dice_loss,
                                                            iou.__name__: iou,
                                                            dice.__name__: dice})
    show_predictions(model=best_model, sample_images=samples)


if __name__ == '__main__':
    main()
