from flask import Flask, render_template, request
import cv2
import os

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def get_students():
    student_list = []
    dataset_path = "dataset"

    if os.path.exists(dataset_path):
        for name in os.listdir(dataset_path):
            path = os.path.join(dataset_path, name)
            if os.path.isdir(path):
                count = len(os.listdir(path))
                student_list.append({
                    "name": name,
                    "count": count
                })
    return student_list


@app.route("/")
def home():
    return render_template(
        "register.html",
        students=get_students()
    )

@app.route("/register", methods=["POST"])
def register():

    name = request.form["name"]
    folder_path = "dataset/" + name

    # ❗ Check if already registered
    if os.path.exists(folder_path):
        return render_template(
            "register.html",
            error="Student already registered!",
            students=get_students()
        )

    os.makedirs(folder_path)

    cam = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            count +=1
            cv2.imwrite(
                folder_path + "/" + str(count) + ".jpg",
                gray[y:y+h,x:x+w]
            )
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow("Capture",img)

        if count>=20:
            break

    cam.release()
    cv2.destroyAllWindows()

    return render_template(
        "dashboard.html",
        name=name,
        total=count
    )

if __name__ == "__main__":
    app.run(debug=True)