import tensorflow as tf
import matplotlib.pyplot as plt

from glob import glob
from utils.loss_and_metrics import iou, dice, JaccardLoss
from utils.configuaration import config_data_pipeline_performance, read_yaml_file
from utils.image_processing import parse_images


def create_test_dataset(data_path: str) -> tuple:
    """
    Creates test dataset.

    :param data_path: path data
    :return: dataset tuple and its size
    """
    dataset_size = len(glob(data_path + "image/*.png"))

    test_dataset = tf.data.Dataset.list_files(data_path + "image/*.png", seed=SEED)
    test_dataset = test_dataset.map(parse_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = {"test": test_dataset}
    dataset['test'] = config_data_pipeline_performance(dataset['test'], False, BUFFER_SIZE, BATCH_SIZE, SEED, AUTOTUNE)

    return dataset, dataset_size


def evaluate_model(model: tf.keras.models.Model, dataset: dict, data_size: int, batch_size: int) -> None:
    """
    Evaluates the model.

    :param model: model to evaluate
    :param dataset: test dataset
    :param data_size: size of test dataset
    :param batch_size: batch size
    """
    test_steps = tf.floor(data_size / batch_size)
    loss, iou_score, dice_score = model.evaluate(dataset['test'], steps=test_steps)

    print(f"Loss: {loss}")
    print(f"IOU: {iou_score}")
    print(f"Dice: {dice_score}")


def show_predictions(model: tf.keras.Model, images: tuple) -> None:
    """
    Shows image, ground truth mask and predicted mask for given images.

    :param model: model for predictions
    :param images: images for predictions
    """
    titles = ['Input image', 'Pos', 'Neg', 'True mask', 'Prediction mask']

    for image, mask in images:
        pred_mask = model.predict(image)
        images_list = [image[0][:, :, :3], image[0][:, :, 3], image[0][:, :, 4], mask[0], pred_mask[0]]

        plt.figure(figsize=(12, 12))
        for i in range(len(images_list)):
            plt.subplot(1, len(images_list), i + 1)
            plt.title(titles[i])
            if i == 1 or i == 2:
                plt.imshow(images_list[i])
            else:
                plt.imshow(tf.keras.preprocessing.image.array_to_img(images_list[i]))
            plt.axis('off')
        plt.show()


if __name__ == '__main__':
    config = read_yaml_file("./config.yaml")

    PROJECT_PATH = config["PROJECT_PATH"]
    DATA_PATH_TEST = PROJECT_PATH + config["DATA_PATH_TEST"]
    MODELS_PATH = PROJECT_PATH + config["MODELS_PATH"]
    UNET_MODEL_PATH = config["UNET_MODEL_PATH"]
    UNET_DC_MODEL_PATH = config["UNET_DC_MODEL_PATH"]
    UNET_DP_MODEL_PATH = config["UNET_DP_MODEL_PATH"]

    BATCH_SIZE = config["BATCH_SIZE"]
    BUFFER_SIZE = config["BUFFER_SIZE"]
    SEED = config["SEED"]
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    SHALLOW_UNET = config["UNET_SHALLOW"]
    UNET_DC = config["UNET_DC"]
    UNET_DP = config["UNET_DP"]

    segmentation_dataset, test_size = create_test_dataset(DATA_PATH_TEST)
    test_images = segmentation_dataset['test'].take(test_size)
    custom_objects = {JaccardLoss.__name__: JaccardLoss(),
                      iou.__name__: iou,
                      dice.__name__: dice}

    if SHALLOW_UNET:
        model_path = MODELS_PATH + UNET_MODEL_PATH
        best_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    elif UNET_DC:
        model_path = MODELS_PATH + UNET_DC_MODEL_PATH
        best_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    elif UNET_DP:
        model_path = MODELS_PATH + UNET_DP_MODEL_PATH
        best_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    else:
        raise NotImplementedError()

    best_model.compile(loss=JaccardLoss(), optimizer="Adam", metrics=[iou, dice])

    evaluate_model(model=best_model,
                   dataset=segmentation_dataset,
                   data_size=test_size,
                   batch_size=BATCH_SIZE)

    show_predictions(model=best_model, images=test_images)
