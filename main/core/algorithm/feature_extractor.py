import numpy as np

from main.core.openpose.openpose import OpenPose


class FeatureExtractor:
    def __init__(self):
        self.open_pose = OpenPose()

    def extract(self, img_sequence):
        feature = []

        for img in img_sequence:
            frame = []

            left_hand, right_hand = self.open_pose.get_hand_key_points_from_image(img)

            # Remove the confidence score from result
            left_hand = np.delete(np.array(left_hand), 2, 1)
            right_hand = np.delete(np.array(right_hand), 2, 1)

            frame.extend(left_hand)
            frame.extend(right_hand)

            feature.append(frame)

        feature = np.array(feature)
        feature = feature.reshape(feature.shape[0], -1)

        return feature


