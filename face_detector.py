import cv2
import argparse



face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def face_detector(path:str = None):


    if path is None:
        camera = cv2.VideoCapture(1)
    else:
        camera = cv2.VideoCapture(path)
    while True:

        ret, frame = camera.read()
        if not ret and path:
            break

        frame = cv2.resize(frame, (850, 520))
        if path is None:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 9,
            minSize = (60, 60)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame,
                (x,y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )
        cv2.imshow("Video",frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to video", nargs="?", default=None)
    parse = parser.parse_args()
    path = parse.path

    face_detector(path=path)