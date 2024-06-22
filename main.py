import pathlib
import cv2 as cv

cascade_path = pathlib.Path(cv.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
print(cascade_path)

classifier = cv.CascadeClassifier(str(cascade_path))

camera = cv.VideoCapture(1)

while True:
    _, frame = camera.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv.CASCADE_SCALE_IMAGE
    )

    for (x,y,h,w) in faces:
        cv.rectangle(frame, (x,y),(x+w,y+h),(225,225,0),2)

    cv.imshow("Faces", frame)
    if cv.waitKey(1) == ord("q"):
        break

camera.release()
cv.destroyAllWindows()

