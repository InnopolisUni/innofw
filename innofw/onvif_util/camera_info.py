import fire
from onvif import ONVIFCamera
from typing import Optional


def get_camera_info(ip, port, user: Optional[str], password: Optional[str]):
    # Создаём объект с указанием хоста, порта, пользователя, пароля и пути до wsdl
    user = str(user)
    password = str(password)
    print(ip, port, "U: " + user, "P: " + password)
    mycam = ONVIFCamera(ip, port, user, password)

    # запрашиваем и выводим информацию об устройстве
    resp = mycam.devicemgmt.GetDeviceInformation()
    print(str(resp))

    # запрашиваем и выводим информацию о сетевых интерфейсах
    resp = mycam.devicemgmt.GetNetworkInterfaces()
    print(str(resp))

    # запрашиваем адрес медиа потока
    media_service = mycam.create_media_service()
    profiles = media_service.GetProfiles()
    token = profiles[0].token
    mycam = media_service.create_type("GetStreamUri")
    mycam.ProfileToken = token
    mycam.StreamSetup = {"Stream": "RTP-Unicast", "Transport": {"Protocol": "RTSP"}}
    print(media_service.GetStreamUri(mycam))


if __name__ == "__main__":
    fire.Fire(get_camera_info)
