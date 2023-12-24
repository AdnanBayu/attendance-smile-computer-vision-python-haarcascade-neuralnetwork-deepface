import cv2
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

class Attendance:
    def __init__(self, foto: str="Captured.png", db: str="Test.jpg"):
        self.img_name = foto
        self.database = db
        self.nimuser = None
        self.result = None
        self.condition = None
        self.attendance = False

    def captureimage(self):
        cam = cv2.VideoCapture(0)
        while True:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            frame = cv2.flip(frame,1)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for x,y,w,h in face:
                img = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)

            cv2.imshow("Space to capture, Esc to close", frame)

            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                cv2.imwrite(self.nimuser+self.img_name, frame)
                print("{} written!".format(self.nimuser+self.img_name))

        cam.release()
        cv2.destroyAllWindows()

    def checkFace(self):
        input1 = self.nimuser+self.img_name
        cek1 = "database/"+self.nimuser+self.database
        self.result = DeepFace.verify(input1, cek1, enforce_detection=False, model_name="VGG-Face",distance_metric="euclidean")
        self.result = self.result["verified"]

    def checkHappy(self):
        input2 = self.nimuser+self.img_name
        self.condition = DeepFace.analyze(input2, enforce_detection=False, actions="emotion")
        self.condition = self.condition[0]["dominant_emotion"]

    def process(self):
        self.login()
        self.captureimage()
        self.checkFace()
        self.checkHappy()
        if self.result==True and self.condition=="happy":
            self.attendance = True
        else:
            self.attendance = False
        print("verify : ", self.result)
        print("emotion : ", self.condition)
        print("Result : ", self.attendance)
        # print(self.attendance)

    def login(self):
        self.nimuser = input("Enter NIM: ")

attendance = Attendance()
attendance.process()