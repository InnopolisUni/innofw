import pytest

from omegaconf import OmegaConf
from typing import Optional
from innofw.onvif_util.stream import show_stream
import unittest
from unittest.mock import patch, Mock

cv2 = pytest.importorskip("cv2")


class TestShowStream(unittest.TestCase):

    @patch('cv2.VideoCapture')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    def test_show_stream(self, mock_wait_key, mock_imshow, mock_video_capture):
        # Mocking the VideoCapture instance
        mock_capture_instance = Mock()
        mock_video_capture.return_value = mock_capture_instance

        # Mocking the return value of read method
        mock_capture_instance.read.side_effect = [(True, Mock()), (False, None)]

        # Call your function with the mocked methods
        uri = 'your_video_uri'
        show_stream(uri)

        # Assertions
        mock_video_capture.assert_called_once_with(uri, cv2.CAP_FFMPEG)
        mock_imshow.assert_called()
        mock_wait_key.assert_called_with(1)

    
