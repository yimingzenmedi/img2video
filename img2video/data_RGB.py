import os
from dataset_RGB import DataLoaderTrain, DataLoaderVal, DataLoaderTest


def get_training_data(rgb_dir, img_options, group_size, pic_index):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, group_size=group_size, pic_index=pic_index, img_options=img_options)


def get_validation_data(rgb_dir, img_options, group_size, pic_index):
    # print("!! get_validation_data")
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, group_size=group_size, pic_index=pic_index, img_options=img_options)


def get_test_data(rgb_dir, img_options):
    print(rgb_dir, img_options)
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)
