import fire
import cv2


def show_stream(uri):
    vcap = cv2.VideoCapture(uri, cv2.CAP_FFMPEG)
    while 1:
        ret, frame = vcap.read()
        if not ret:
            print("Frame is empty")
            break
        else:
            cv2.imshow("VIDEO", frame)
            cv2.waitKey(1)


if __name__ == "__main__":
    fire.Fire(show_stream)
