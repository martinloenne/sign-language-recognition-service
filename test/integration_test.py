import unittest
import base64
import cv2

from flask_testing import TestCase
from flask import json

from main.common.utils import extract_frames_from_video
from app import app


# Converts a base64 bytes file to its blob string representation
def convert_base64_to_string_format(b64, type):
    b64_string = str(b64)
    return f'data:{type};base64,{b64_string[2:len(b64_string)-1]}'


class TestIntegration(TestCase):
    def create_app(self):
        return app

    # Test whether a status code 200 is returned when a valid Base64 image array is sent with request
    def test_endpoint_valid_data_returns_status200(self):
        frames = extract_frames_from_video('./test_data/valid_data.mp4')

        valid_base64_data = []
        for frame in frames:
            ret, buffer = cv2.imencode('.jpg', frame)
            b64 = base64.b64encode(buffer)
            valid_base64_data.append(convert_base64_to_string_format(b64, 'image/jpg'))

        response = app.test_client().post('/recognize', data=json.dumps(valid_base64_data))

        self.assert200(response)

    # Test whether a status code 400 is returned when invalid non-Base64 data is sent with request
    def test_endpoint_invalid_non_base64_data_returns_status400(self):
        invalid_non_base64_data = 0

        response = app.test_client().post('/recognize', data=json.dumps(invalid_non_base64_data))

        self.assert400(response)

    # Test whether a status code 400 is returned when null value is sent with request
    def test_endpoint_none_data_returns_status400(self):
        response = app.test_client().post('/recognize', data=json.dumps(None))

        self.assert400(response)

    # Test whether a status code 400 is returned when invalid Base64 data is sent with request
    def test_endpoint_invalid_base64_data_returns_status400(self):
        in_file = open("./test_data/invalid_data", "rb")
        b64 = base64.b64encode(in_file.read())
        invalid_base64_data = convert_base64_to_string_format(b64, 'image/jpg')
        in_file.close()

        response = app.test_client().post('/recognize', data=json.dumps([invalid_base64_data]))

        self.assert400(response)


if __name__ == '__main__':
    unittest.main()