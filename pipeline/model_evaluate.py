import os.path
import tensorflow as tf
import matplotlib.pyplot as plt

from glob import glob
from utils.loss_and_metrics import iou, dice, JaccardLoss
from utils.configuaration import config_data_pipeline_performance, read_yaml_file
from utils.image_processing import parse_images

from unet_architectures.ShallowUnet import ShallowUnet
from unet_architectures.DualPathUnet import DualPathUnet
from unet_architectures.AttentionDualPath import AttentionDualPathUnet


def create_test_dataset(data_path: str) -> tuple:
    """
    Creates test dataset.

    :param data_path: path data
    :return: dataset tuple and its size
    """
    dataset_size = len(glob(data_path + "image/*.png"))
    autotune = tf.data.experimental.AUTOTUNE

    test_dataset = tf.data.Dataset.list_files(data_path + "image/*.png", seed=seed)
    test_dataset = test_dataset.map(parse_images, num_parallel_calls=autotune)

    dataset = {"test": test_dataset}
    dataset['test'] = config_data_pipeline_performance(dataset['test'], False, batch_size, batch_size, seed, autotune)

    return dataset, dataset_size


def evaluate_model(model: tf.keras.models.Model, dataset: dict, data_size: int, batch: int) -> None:
    """
    Evaluates the model.

    :param model: model to evaluate
    :param dataset: test dataset
    :param data_size: size of test dataset
    :param batch: batch size
    """
    test_steps = tf.floor(data_size / batch)
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

    project_dir = config["project_dir"]
    test_data_dir = project_dir + config["test_data_dir"]
    models_dir = config["models_dir"]

    batch_size = config["batch_size"]
    buffer_size = config["buffer_size"]
    seed = config["seed"]

    shallow = config["shallow"]
    dual_path = config["dual_path"]
    attention_dual_path = config["attention_dual_path"]

    segmentation_dataset, test_size = create_test_dataset(test_data_dir)
    test_images = segmentation_dataset['test'].take(test_size)
    custom_objects = {JaccardLoss.__name__: JaccardLoss(),
                      iou.__name__: iou,
                      dice.__name__: dice}

    if shallow:
        model_path = os.path.join(models_dir, ShallowUnet.__name__ + '.hdf5')
        best_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    elif dual_path:
        model_path = os.path.join(models_dir, DualPathUnet.__name__ + '.hdf5')
        best_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    elif attention_dual_path:
        model_path = os.path.join(models_dir, AttentionDualPathUnet.__name__ + '.hdf5')
        best_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    else:
        raise NotImplementedError()

    best_model.compile(loss=JaccardLoss(), optimizer=tf.optimizers.Adam(), metrics=[iou, dice])

    evaluate_model(model=best_model,
                   dataset=segmentation_dataset,
                   data_size=test_size,
                   batch=batch_size)
