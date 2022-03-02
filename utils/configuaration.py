import yaml as yaml
import tensorflow as tf


def read_yaml_file(yaml_file_path: str):
    """
    A helper function for reading yaml files.

    :param yaml_file_path: path to the yaml file
    :return: content of the yaml file
    """
    with open(yaml_file_path, 'r') as stream:
        try:
            content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise exc
    return content


def config_data_pipeline_performance(dataset: tf.data.Dataset, shuffle: bool, buffer_size: int, batch_size: int,
                                     seed: int) -> tf.data.Dataset:
    """
    Configures the dataset processing pipeline for its performance enhancement.

    :param dataset: dataset to be configured
    :param shuffle: flag indicating if to apply dataset shuffling
    :param buffer_size: size of the buffer
    :param batch_size: size of the batch
    :param seed: random seed
    :return: Configured dataset
    """
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
