import unittest
from unittest.mock import patch, MagicMock
import cv2

from stream import show_stream

class TestShowStream(unittest.TestCase):

    @patch('cv2.VideoCapture')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    def test_show_stream(self, mock_waitkey, mock_imshow, mock_videocapture):
        mock_vcap = MagicMock(spec=cv2.VideoCapture)
        mock_videocapture.return_value = mock_vcap

        mock_vcap.read.return_value = (True, MagicMock())
        mock_waitkey.return_value = 27  # Escape key

        show_stream('fake_uri')

        mock_videocapture.assert_called_once_with('fake_uri', cv2.CAP_FFMPEG)
        mock_vcap.read.assert_called()
        mock_imshow.assert_called()
        mock_waitkey.assert_called_once_with(1)

if __name__ == '__main__':
    unittest.main()
