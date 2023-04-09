#path_onvif = Users.randalllionelkharkrang.Desktop.Innopolis.innofw.innofw.onvif_util
import camera_info
import unittest
from unittest.mock import patch
from camera_info import get_camera_info

class TestGetCameraInfo(unittest.TestCase):

    @patch('camera_info.ONVIFCamera')
    @patch('builtins.print')
    def test_get_camera_info(self, mock_print, mock_onvif_camera):
        # Mock input parameters
        ip = '192.168.1.1'
        port = 8080
        user = 'username'
        password = 'password'

        # Mock ONVIFCamera instance and its methods
        mock_devicemgmt = mock_onvif_camera.return_value.devicemgmt
        mock_devicemgmt.GetDeviceInformation.return_value = 'Device Information Response'
        mock_devicemgmt.GetNetworkInterfaces.return_value = 'Network Interfaces Response'
        mock_media_service = mock_onvif_camera.return_value.create_media_service.return_value
        mock_profiles = mock_media_service.GetProfiles.return_value
        mock_profiles[0].token = 'profile_token'
        mock_media_service.create_type.return_value = mock_profiles
        mock_media_service.GetStreamUri.return_value = 'Stream URI'

        # Call the function
        get_camera_info(ip, port, user, password)

        # Assert that the expected methods are called with the expected arguments
        mock_onvif_camera.assert_called_once_with(ip, port, user, password)
        mock_devicemgmt.GetDeviceInformation.assert_called_once()
        mock_print.assert_any_call('Device Information Response')
        mock_devicemgmt.GetNetworkInterfaces.assert_called_once()
        mock_print.assert_any_call('Network Interfaces Response')
        mock_media_service.GetProfiles.assert_called_once()
        mock_media_service.create_type.assert_called_once_with('GetStreamUri')
        mock_print.assert_any_call('Stream URI')

if __name__ == '__main__':
    unittest.main()
