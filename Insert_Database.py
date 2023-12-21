import cv2

class Database:
    def __init__(self, db: str="Test.jpg"):
        self.database = db
        self.nimuser = None

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
                cv2.imwrite("database/"+self.nimuser+self.database, frame)
                print("{} written!".format("database/"+self.nimuser+self.database))

        cam.release()
        cv2.destroyAllWindows()

    def login(self):
        self.nimuser = input("Enter NIM: ")

database = Database()
database.login()
database.captureimage()