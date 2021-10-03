import yaml as yaml
import tensorflow as tf


def read_yaml_file(yaml_file_path: str):
    """
    Helper function for reading yaml files.

    :param yaml_file_path: path to yaml file
    :return: content of opened yaml file
    """
    with open(yaml_file_path, 'r') as stream:
        try:
            content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise exc
    return content


def config_data_pipeline_performance(dataset: tf.data.Dataset, training: bool, buffer_size: int, batch_size: int,
                                     seed: int, autotune: int) -> tf.data.Dataset:
    """
    Configure the utils pipeline for its performance enhancement.

    :param dataset: dataset to be configured
    :param training: a boolean which if true indicates that the dataset set is the training one
    :param buffer_size: size of the buffer
    :param batch_size: size of the batch
    :param seed: random seed for creation of the distribution
    :param autotune: maximum number of elements that will be used when prefetching
    :return: Configured dataset
    """
    if training:
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=autotune)

    return dataset
