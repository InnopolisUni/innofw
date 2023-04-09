import unittest
from unittest.mock import patch, MagicMock
import cv2
from stream import show_stream

class TestShowStream(unittest.TestCase):

    @patch('cv2.VideoCapture')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('builtins.print')
    def test_show_stream(self, mock_print, mock_cv2_waitKey, mock_cv2_imshow, mock_cv2_video_capture):
        # Mock input parameter
        uri = 'rtsp://example.com/stream'

        # Mock VideoCapture instance and its methods
        mock_vcap = mock_cv2_video_capture.return_value
        mock_vcap.read.side_effect = [(True, 'frame1'), (True, 'frame2'), (False, None)]

        # Call the function
        show_stream(uri)

        # Assert that the expected methods are called with the expected arguments
        mock_cv2_video_capture.assert_called_once_with(uri, cv2.CAP_FFMPEG)
        self.assertEqual(mock_vcap.read.call_count, 3)
        mock_cv2_imshow.assert_called_with('VIDEO', 'frame2')
        #mock_cv2_waitKey.assert_called_once_with(1)
        self.assertEqual(mock_print.call_count, 1)
        mock_print.assert_called_with('Frame is empty')
if __name__ == '__main__':
    unittest.main()
