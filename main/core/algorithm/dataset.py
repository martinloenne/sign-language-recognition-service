import random
import pickle
import numpy as np
import os

from main.core.algorithm.feature_extractor import FeatureExtractor
from main.common.utils import Timer, calc_percentage, extract_frames_from_video, log, warning
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


class DataSet:
    def __init__(self, train, val, test, labels, window, stride, feature_size):
        self.train = train
        self.val = val
        self.test = test
        self.labels = labels
        self.window = window
        self.stride = stride
        self.feature_size = feature_size

    def save(self, name, save_dir):
        pickle_out = open(f"{save_dir}/{name}.pickle", "wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()


class DataSetCreator:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()

    def create_dataset(self, raw_data, labels, window_length, stride, batch_size, val_split, test_split):
        # Shuffle data for train, val and test split
        random.shuffle(raw_data)

        log(f'Creating {1-(val_split+test_split)}-{val_split}-{test_split} train-validation-test split.')

        test_index = int(len(raw_data) * test_split)
        val_index = int(len(raw_data) * (test_split + val_split))

        test = raw_data[0:test_index]
        val = raw_data[test_index:val_index]
        train = raw_data[val_index:len(raw_data)]

        log(f'Training set: {len(train)} videos.')
        log(f'Validation set: {len(val)} videos.')
        log(f'Test set: {len(test)} videos.')

        # Sort by label
        test.sort(key=lambda x: x[1])
        val.sort(key=lambda x: x[1])
        train.sort(key=lambda x: x[1])

        features_val, target_val = self._convert_raw_dataset_to_features_and_targets(val)
        features_train, target_train = self._convert_raw_dataset_to_features_and_targets(train)
        features_test, target_test = self._convert_raw_dataset_to_features_and_targets(test)

        ts_generator_val = TimeseriesGenerator(features_val, target_val, length=window_length,
                                               sampling_rate=1, batch_size=batch_size, stride=stride, shuffle=True)
        ts_generator_train = TimeseriesGenerator(features_train, target_train, length=window_length,
                                                 sampling_rate=1, batch_size=batch_size, stride=stride, shuffle=True)
        ts_generator_test = TimeseriesGenerator(features_test, target_test, length=window_length,
                                                sampling_rate=1, batch_size=batch_size, stride=stride, shuffle=True)

        return DataSet(ts_generator_train, ts_generator_val, ts_generator_test,
                       labels, window_length, stride, features_train.shape[1])

    def _convert_raw_dataset_to_features_and_targets(self, raw_data):

        features = []
        targets = []

        # Extract each frame from src videos to feature array and add corresponding label to target array
        for feature, target in raw_data:
            for frame in feature:
                features.append(frame)
                targets.append(target)

        return np.array(features), np.array(targets)

    def create_raw_dataset_from_src(self, src_dir):
        log("Creating raw dataset from source videos.")

        timer = Timer()
        timer.start()

        labels = os.listdir(src_dir)  # Get paths of all labels

        dataset = []
        for label in labels:
            path = f'{src_dir}/{label}'
            label_index = labels.index(label)

            data = self._create_raw_data_from_label_src(label_index, label, path)
            dataset.extend(data)

        log(f'Raw dataset created from source videos. Number of videos: {len(dataset)}.')
        log(f'Total elapsed time: {timer.get_elapsed_time()}.')

        return dataset, labels

    def _create_raw_data_from_label_src(self, label_index, label, path):
        raw_data = []

        video_paths = os.listdir(path)  # Get paths of all src videos for label

        log(f"Creating raw data for '{label}'.")
        for i, video in enumerate(video_paths):
            try:
                sequence = extract_frames_from_video(f'{path}/{video}')
                feature = self.feature_extractor.extract(sequence)
                raw_data.append([feature, label_index])
            except Exception as ex:
                warning(ex)
                warning("Skipping video.")
                pass

            log(f"'{label}' data creation {calc_percentage(i + 1, len(video_paths))}% done.")

        return raw_data

    def save_raw_dataset(self, raw_dataset, dataset_name, labels, save_dir):
        pickle_out = open(f"{save_dir}/{dataset_name}_raw.pickle", "wb")
        pickle.dump(raw_dataset, pickle_out)
        pickle_out.close()

        pickle_out = open(f"{save_dir}/{dataset_name}_labels.pickle", "wb")
        pickle.dump(labels, pickle_out)
        pickle_out.close()

