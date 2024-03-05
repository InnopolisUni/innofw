import pytest


from innofw.onvif_util.mover import CameraControl
import unittest
from unittest.mock import patch, Mock

onvif = pytest.importorskip("onvif")

@pytest.mark.skip(reason="does not work")
class TestCameraControl(unittest.TestCase):
    @patch('onvif.ONVIFCamera')
    def test_absolute_move(self, mock_onvif_camera):
        # Mock the ONVIFCamera class to include wsdl_dir attribute
        mock_onvif_camera_instance = mock_onvif_camera.return_value
        mock_onvif_camera_instance.create_media_service.return_value = Mock()
        mock_onvif_camera_instance.create_ptz_service.return_value = Mock()
        mock_onvif_camera_instance.GetProfiles.return_value = [Mock()]
        mock_onvif_camera_instance.AbsoluteMove.return_value = Mock()
        mock_onvif_camera_instance.ContinuousMove.return_value = Mock()
        mock_onvif_camera_instance.RelativeMove.return_value = Mock()
        mock_onvif_camera_instance.Stop.return_value = Mock()

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

        # Mocking the return value of GetStreamUri
        mock_media_service = Mock()
        mock_token = "mock_token"
        mock_media_service.create_type.return_value = Mock()
        mock_media_service.create_type.return_value.ProfileToken = mock_token

        mock_media_service.GetStreamUri.return_value = "mock_stream_uri"
        
        # Set wsdl_dir attribute for the mock instance
        mock_onvif_camera_instance.wsdl_dir = '/path/to/wsdl'

        # Create an instance of CameraControl
        ip = 'your_ip'
        user = 'your_user'
        password = 'your_password'
        camera_control = CameraControl(ip, user, password)

        # Call absolute_move method
        pan = 0.5
        tilt = 0.3
        zoom = 1.0
        resp = camera_control.absolute_move(pan, tilt, zoom)

        # Assertions
        self.assertIsNotNone(resp)
        mock_onvif_camera.assert_called_once_with(ip, 80, user, password)
        mock_onvif_camera_instance.create_media_service.assert_called_once()
        mock_onvif_camera_instance.create_ptz_service.assert_called_once()
        mock_onvif_camera_instance.GetProfiles.assert_called_once()
        mock_onvif_camera_instance.AbsoluteMove.assert_called_once()

        move = camera_control.continuous_move(pan, tilt, zoom)
        self.assertIsNotNone(move)
        mock_onvif_camera_instance.GetProfiles.assert_called_once()
        mock_onvif_camera_instance.ContinuousMove.assert_called_once()

        stop = camera_control.stop_move()
        self.assertIsNotNone(stop)
        mock_onvif_camera_instance.Stop.assert_called_once()


if __name__ == '__main__':
    unittest.main()

    
