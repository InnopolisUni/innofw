#path_onvif = Users.randalllionelkharkrang.Desktop.Innopolis.innofw.innofw.onvif_util
import camera_info
import unittest
from unittest.mock import patch


class TestCameraInfo(unittest.TestCase):
    @patch('camera_info.ONVIFCamera')
    #@patch('builtins.print')
    def test_get_camera_info(self,mock_onvif_camera):
        # Set up mock ONVIFCamera instance and methods
        mock_device_info = mock_onvif_camera.return_value.devicemgmt.GetDeviceInformation
        mock_device_info.return_value = 'Device Information'
        mock_network_interfaces = mock_onvif_camera.return_value.devicemgmt.GetNetworkInterfaces
        mock_network_interfaces.return_value = 'Network Interfaces'
        mock_profiles = mock_onvif_camera.return_value.create_media_service.return_value.GetProfiles
        mock_profiles.return_value = [unittest.mock.Mock(token='profile_token')]
        mock_stream_uri = mock_onvif_camera.return_value.create_media_service.return_value.GetStreamUri
        mock_stream_uri.return_value = 'rtsp://example.com/stream'
        # Call the function to test
        camera_info.get_camera_info('192.168.1.100', 80, 'admin', 'password')
        # Check if ONVIFCamera class and its methods were called with correct arguments
        mock_onvif_camera.assert_called_once_with('192.168.1.100', 80, 'admin', 'password')
        mock_device_info.assert_called_once_with()
        mock_network_interfaces.assert_called_once_with()
        mock_profiles.assert_called_once_with()
        mock_stream_uri.assert_called_once_with(
            {'ProfileToken': 'profile_token',
             'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}}})

if __name__ == '__main__':
    unittest.main()
