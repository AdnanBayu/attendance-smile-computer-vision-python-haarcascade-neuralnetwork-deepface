import cv2
from deepface import DeepFace

class Attendance:
    def __init__(self, foto: str="imagecaptured.png", db: str="test.jpg"):
        self.img_name = foto
        self.database = db
        self.attendance = False

    def captureimage(self):
        cam = cv2.VideoCapture(0)
        while True:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            frame = cv2.flip(frame,1)
            cv2.imshow("Space to capture, Esc to close", frame)

            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                cv2.imwrite(self.img_name, frame)
                print("{} written!".format(self.img_name))

        cam.release()
        cv2.destroyAllWindows()

    def checkFace(self):
        self.result = DeepFace.verify(self.img_name, self.database, enforce_detection=False, model_name="VGG-Face",distance_metric="euclidean")

    def checkHappy(self):
        self.condition = DeepFace.analyze(self.img_name, enforce_detection=False, actions="emotion")

    def process(self):
        self.captureimage()
        self.checkFace()
        self.checkHappy()
        if self.checkFace()==True and self.checkHappy()=="happy":
            self.attendance = True
        else:
            self.attendance = False
        print("verify : ", self.result["verified"])
        print("emotion : ", self.condition[0]["dominant_emotion"])
        print("Result : ", self.attendance)

attendance = Attendance()
attendance.process()