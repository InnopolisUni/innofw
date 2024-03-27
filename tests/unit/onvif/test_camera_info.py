import pytest


from innofw.onvif_util.camera_info import get_camera_info
import unittest
from unittest.mock import patch, Mock

onvif = pytest.importorskip("onvif")
@pytest.mark.skip(reason="does not work")
class TestGetCameraInfo(unittest.TestCase):

    @patch('onvif.ONVIFCamera', autospec=True)
    def test_get_camera_info(self, mock_onvif_camera):
        # Mocking ONVIFCamera to avoid actual network requests
        mock_camera_instance = Mock()
        mock_onvif_camera.return_value = mock_camera_instance

        # Mocking the update_xaddrs method
        mock_camera_instance.update_xaddrs.return_value = None

        # Mocking the return values of GetDeviceInformation and GetNetworkInterfaces
        mock_device_info = Mock()
        mock_network_interfaces = Mock()
        mock_camera_instance.devicemgmt.GetDeviceInformation.return_value = mock_device_info
        mock_camera_instance.devicemgmt.GetNetworkInterfaces.return_value = mock_network_interfaces
        mock_camera_instance.devicemgmt.GetCapabilities.return_value = [None]

        # Mocking the return value of GetStreamUri
        mock_media_service = Mock()
        mock_profiles = [Mock()]
        mock_media_service.GetProfiles.return_value = mock_profiles
        mock_token = "mock_token"
        mock_media_service.create_type.return_value = Mock()
        mock_media_service.create_type.return_value.ProfileToken = mock_token
        mock_camera_instance.create_media_service.return_value = mock_media_service
        mock_media_service.GetStreamUri.return_value = "mock_stream_uri"

        # Call your function with the mocked ONVIFCamera
        ip = 'your_ip'
        port = 80
        user = 'your_user'
        password = 'your_password'
        get_camera_info(ip, port, user, password)

        # Assertions
        mock_onvif_camera.assert_called_once_with(ip, port, user, password)
        mock_camera_instance.devicemgmt.GetDeviceInformation.assert_called_once()
        mock_camera_instance.devicemgmt.GetNetworkInterfaces.assert_called_once()
        mock_media_service.GetProfiles.assert_called_once()
        mock_media_service.create_type.assert_called_once_with("GetStreamUri")
        mock_media_service.GetStreamUri.assert_called_once()

    
