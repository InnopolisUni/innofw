import unittest
from unittest.mock import patch, Mock
from mover import CameraControl

class TestCameraControl(unittest.TestCase):

    @patch('onvif.ONVIFCamera')
    def test_init(self, mock_onvif_camera):
        # Mocking the ONVIFCamera constructor
        ip = "192.168.0.1"
        user = "admin"
        password = "password"
        camera_control = CameraControl(ip, user, password)
        mock_onvif_camera.assert_called_once_with(ip, 80, user, password)
        assert camera_control.camera_media_profile == mock_onvif_camera.return_value.create_media_service.return_value.GetProfiles.return_value[0]
        assert camera_control.camera_ptz == mock_onvif_camera.return_value.create_ptz_service.return_value

    @patch.object(CameraControl, '_map_onvif_to_vapix')
    @patch.object(CameraControl, '_map_vapix_to_onvif')
    def test_absolute_move(self, mock_map_vapix_to_onvif, mock_map_onvif_to_vapix):
        # Mocking the create_type and AbsoluteMove methods of the camera_ptz object
        camera_control = CameraControl("192.168.0.1", "admin", "password")
        mock_onvif_response = Mock()
        camera_control.camera_ptz.AbsoluteMove.return_value = mock_onvif_response

        # Mocking the map functions to set the expected values
        mock_map_vapix_to_onvif.return_value = 10.0
        mock_map_onvif_to_vapix.return_value = 20.0

        # Testing the method
        pan, tilt, zoom = 1.0, 2.0, 3.0
        actual_response = camera_control.absolute_move(pan, tilt, zoom)

        # Assertions
        mock_map_vapix_to_onvif.assert_called_once_with(pan, -1.0, 1.0, -1.0, 1.0)
        mock_map_onvif_to_vapix.assert_called_once_with(pan, -1.0, 1.0, -1.0, 1.0)
        camera_control.camera_ptz.create_type.assert_called_once_with('AbsoluteMove')
        assert actual_response == mock_onvif_response
        assert camera_control.camera_ptz.AbsoluteMove.return_value == mock_onvif_response
        assert camera_control.camera_ptz.create_type.return_value.Position == {'PanTilt': {'x': 10.0, 'y': 20.0}, 'Zoom': zoom}

    @patch.object(CameraControl, '_map_onvif_to_vapix')
    @patch.object(CameraControl, '_map_vapix_to_onvif')
    def test_continuous_move(self, mock_map_vapix_to_onvif, mock_map_onvif_to_vapix):
        # Mocking the create_type and ContinuousMove methods of the camera_ptz object
        camera_control = CameraControl("192.168.0.1", "admin", "password")
        mock_onvif_response = Mock()
        camera_control.camera_ptz.ContinuousMove.return_value = mock_onvif_response

        # Mocking the map functions to set the expected values
        mock_map_vapix_to_onvif.return_value = 10.0
        mock_map_onvif_to_vapix.return_value = 20.0

        # Testing the method
        pan, tilt, zoom = 1.0, 2

if __name__ == '__main__':
    unittest.main()
