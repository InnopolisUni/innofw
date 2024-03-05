import logging
from typing import Optional

import fire
from onvif import ONVIFCamera


class CameraControl:
    """
    A class to represent a Camera unit with ONVIF support, with corresponding methods for control of movements

    ...

    Attributes
    ----------
    ip : str
    user : str
    password: str


    Methods
    -------
    absolute_move(self, pan: float, tilt: float, zoom: float):
        Operation to move pan, tilt or zoom to a absolute destination.
    continuous_move(self, pan: float, tilt: float, zoom: float):
        Operation for continuous Pan/Tilt and Zoom movements.
    relative_move(self, pan: float, tilt: float, zoom: float):
        Operation for Relative Pan/Tilt and Zoom Move.
    stop_move(self):
        Operation to stop ongoing pan, tilt and zoom movements of absolute relative and continuous type.
    set_home_position(self):
        Operation to save current position as the home position.
    go_home_position(self):
        Operation to move the PTZ device to it's "home" position.
    get_ptz(self):
        Operation to request PTZ status.
    set_preset(self, preset_name: str):
        The command saves the current device position parameters.
    remove_preset(self, preset_name: str):
        Operation to remove a PTZ preset.
    go_to_preset(self, preset_position: str):
        Operation to go to a saved preset position.
    """

    def __init__(self, ip, user, password):
        self.__cam_ip = ip
        self.__cam_user = str(user)
        self.__cam_password = str(password)

        mycam = ONVIFCamera(
            self.__cam_ip, 80, self.__cam_user, self.__cam_password, wsdl_dir='./innofw/onvif_util/wsdl'
        )
        logging.info("Create media service object")
        media = mycam.create_media_service()
        logging.info("Create ptz service object")
        ptz = mycam.create_ptz_service()
        logging.info("Get target profile")
        media_profile = media.GetProfiles()[0]
        logging.info("Camera working!")

        self.mycam = mycam
        self.camera_ptz = ptz
        self.camera_media_profile = media_profile
        self.camera_media = media

    @staticmethod
    def _map_onvif_to_vapix(value, min_onvif, max_onvif, min_vapix, max_vapix):
        return (value - min_onvif) * (max_vapix - min_vapix) / (
            max_onvif - min_onvif
        ) + min_vapix

    @staticmethod
    def _map_vapix_to_onvif(value, min_vapix, max_vapix, min_onvif, max_onvif):
        return (value - min_vapix) * (max_onvif - min_onvif) / (
            max_vapix - min_vapix
        ) + min_onvif

    def absolute_move(self, pan: float, tilt: float, zoom: float):
        """
        Operation to move pan, tilt or zoom to a absolute destination.
        Args:
            pan: Pans the device relative to the (0,0) position.
            tilt: Tilts the device relative to the (0,0) position.
            zoom: Zooms the device n steps.
        Returns:
            Return onvif's response
        """
        request = self.camera_ptz.create_type("AbsoluteMove")
        request.ProfileToken = self.camera_media_profile.token
        request.Position = {"PanTilt": {"x": pan, "y": tilt}, "Zoom": zoom}
        resp = self.camera_ptz.AbsoluteMove(request)
        logging.info(
            "camera_command( aboslute_move(%f, %f, %f) )", pan, tilt, zoom
        )
        return resp

    def continuous_move(self, pan: float, tilt: float, zoom: float):
        """
        Operation for continuous Pan/Tilt and Zoom movements.
        Args:
            pan: speed of movement of Pan.
            tilt: speed of movement of Tilt.
            zoom: speed of movement of Zoom.
        Returns:
            Return onvif's response.
        """
        request = self.camera_ptz.create_type("ContinuousMove")
        request.ProfileToken = self.camera_media_profile.token
        request.Velocity = {"PanTilt": {"x": pan, "y": tilt}, "Zoom": zoom}
        resp = self.camera_ptz.ContinuousMove(request)
        logging.info(
            "camera_command( continuous_move(%f, %f, %f) )", pan, tilt, zoom
        )
        return resp

    def relative_move(self, pan: float, tilt: float, zoom: float):
        """
        Operation for Relative Pan/Tilt and Zoom Move.
        Args:
            pan: A positional Translation relative to the pan current position.
            tilt: A positional Translation relative to the tilt current position.
            zoom:
        Returns:
            Return onvif's response
        """
        request = self.camera_ptz.create_type("RelativeMove")
        request.ProfileToken = self.camera_media_profile.token
        request.Translation = {"PanTilt": {"x": pan, "y": tilt}, "Zoom": zoom}
        resp = self.camera_ptz.RelativeMove(request)
        logging.info(
            "camera_command( relative_move(%f, %f, %f) )", pan, tilt, zoom
        )
        return resp

    def stop_move(self):
        """
        Operation to stop ongoing pan, tilt and zoom movements of absolute relative and continuous type.
        Returns:
            Return onvif's response
        """
        request = self.camera_ptz.create_type("Stop")
        request.ProfileToken = self.camera_media_profile.token
        resp = self.camera_ptz.Stop(request)
        logging.info("camera_command( stop_move() )")
        return resp

    def set_home_position(self):
        """
        Operation to save current position as the home position.
        Returns:
            Return onvif's response
        """
        request = self.camera_ptz.create_type("SetHomePosition")
        request.ProfileToken = self.camera_media_profile.token
        resp = self.camera_ptz.SetHomePosition(request)
        self.camera_ptz.Stop({"ProfileToken": self.camera_media_profile.token})
        logging.info("camera_command( set_home_position() )")
        return resp

    def go_home_position(self):
        """
        Operation to move the PTZ device to it's "home" position.
        Returns:
            Return onvif's response
        """
        request = self.camera_ptz.create_type("GotoHomePosition")
        request.ProfileToken = self.camera_media_profile.token
        resp = self.camera_ptz.GotoHomePosition(request)
        logging.info("camera_command( go_home_position() )")
        return resp

    def get_ptz(self):
        """
        Operation to request PTZ status.
        Returns:
            Returns a list with the values ​​of Pan, Tilt and Zoom
        """
        request = self.camera_ptz.create_type("GetStatus")
        request.ProfileToken = self.camera_media_profile.token
        ptz_status = self.camera_ptz.GetStatus(request)
        pan = ptz_status.Position.PanTilt.x
        tilt = ptz_status.Position.PanTilt.y
        zoom = ptz_status.Position.Zoom.x
        ptz_list = (pan, tilt, zoom)
        logging.info("camera_command( get_ptz() )")
        return ptz_list

    def set_preset(self, preset_name: str):
        """
        The command saves the current device position parameters.
        Args:
            preset_name: Name for preset.
        Returns:
            Return onvif's response.
        """
        presets = CameraControl.get_preset_complete(self)
        request = self.camera_ptz.create_type("SetPreset")
        request.ProfileToken = self.camera_media_profile.token
        request.PresetName = preset_name
        logging.info("camera_command( set_preset%s) )", preset_name)

        for i, _ in enumerate(presets):
            if str(presets[i].Name) == preset_name:
                logging.warning(
                    "Preset ('%s') not created. Preset already exists!",
                    preset_name,
                )
                return None

        ptz_set_preset = self.camera_ptz.SetPreset(request)
        logging.info("Preset ('%s') created!", preset_name)
        return ptz_set_preset

    def get_preset(self):
        """
        Operation to request all PTZ presets.
        Returns:
            Returns a list of tuples with the presets.
        """
        ptz_get_presets = CameraControl.get_preset_complete(self)
        logging.info("camera_command( get_preset() )")

        presets = []
        for i, _ in enumerate(ptz_get_presets):
            presets.append((i, ptz_get_presets[i].Name))
        return presets

    def get_preset_complete(self):
        """
        Operation to request all PTZ presets.
        Returns:
            Returns the complete presets Onvif.
        """
        request = self.camera_ptz.create_type("GetPresets")
        request.ProfileToken = self.camera_media_profile.token
        ptz_get_presets = self.camera_ptz.GetPresets(request)
        return ptz_get_presets

    def remove_preset(self, preset_name: str):
        """
        Operation to remove a PTZ preset.
        Args:
            preset_name: Preset name.
        Returns:
            Return onvif's response.
        """
        presets = CameraControl.get_preset_complete(self)
        request = self.camera_ptz.create_type("RemovePreset")
        request.ProfileToken = self.camera_media_profile.token
        logging.info("camera_command( remove_preset(%s) )", preset_name)
        for i, _ in enumerate(presets):
            if str(presets[i].Name) == preset_name:
                request.PresetToken = presets[i].token
                ptz_remove_preset = self.camera_ptz.RemovePreset(request)
                logging.info("Preset ('%s') removed!", preset_name)
                return ptz_remove_preset
        logging.warning("Preset ('%s') not found!", preset_name)
        return None

    def go_to_preset(self, preset_position: str):
        """
        Operation to go to a saved preset position.
        Args:
            preset_position: preset name.
        Returns:
            Return onvif's response.
        """
        presets = CameraControl.get_preset_complete(self)
        request = self.camera_ptz.create_type("GotoPreset")
        request.ProfileToken = self.camera_media_profile.token
        logging.info("camera_command( go_to_preset(%s) )", preset_position)
        for i, _ in enumerate(presets):
            str1 = str(presets[i].Name)
            if str1 == preset_position:
                request.PresetToken = presets[i].token
                resp = self.camera_ptz.GotoPreset(request)
                logging.info("Goes to ('%s')", preset_position)
                return resp
        logging.warning("Preset ('%s') not found!", preset_position)
        return None



def move(ip, user: Optional[str], password: Optional[str], move_type):
    """
    Move camera according to move type

    Args:
        ip (str):
        user (str):
        password (str):
        move_type (str): Supported moves are:  zoom_in, zoom_out, pan_left, pan_right, tilt_up, tilt_down
    """
    logging.basicConfig(
        filename="teste-onvif.log", filemode="w", level=logging.DEBUG
    )
    logging.info("Started")

    ptz_cam = CameraControl(ip, user, password)

    if move_type == "zoom_in":
        ptz_cam.relative_move(0.0, 0.0, 0.5)
    elif move_type == "zoom_out":
        ptz_cam.relative_move(0.0, 0.0, -0.5)
    elif move_type == "pan_left":
        ptz_cam.relative_move(-0.5, 0.0, 0.0)
    elif move_type == "pan_right":
        ptz_cam.relative_move(0.5, 0.0, 0.0)
    elif move_type == "tilt_up":
        ptz_cam.relative_move(0.0, 0.5, 0.0)
    elif move_type == "tilt_down":
        ptz_cam.relative_move(0.0, -0.5, 0.0)


if __name__ == "__main__":
    fire.Fire(move)
