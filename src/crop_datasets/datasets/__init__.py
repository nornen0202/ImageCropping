from crop_datasets.datasets.base import DatasetManager


def manager_for(info):
    return DatasetManager(info)
