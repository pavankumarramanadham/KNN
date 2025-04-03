import cv2
import os
from flask import Flask, request, render_template, jsonify
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
from flask_sqlalchemy import SQLAlchemy
import base64
import io
from PIL import Image

# Defining Flask App
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Secure random key should be used in production

nimgs = 10

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initializing CascadeClassifier object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# If these directories don't exist, create them
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

# Configuring SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Classes Model
class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)
    roll = db.Column(db.Integer)
    time = db.Column(db.DateTime)
    date = db.Column(db.Date)

# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

# extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

# Extract info from today's attendance using SQLAlchemy
def extract_attendance():
    today_date = date.today()
    attendance_records = Attendance.query.filter_by(date=today_date).all()
    names = [record.name for record in attendance_records]
    rolls = [record.roll for record in attendance_records]
    times = [record.time.strftime("%H:%M:%S") for record in attendance_records]
    l = len(attendance_records)
    return names, rolls, times, l

# Add Attendance of a specific user using SQLAlchemy
def add_attendance(name):
    username = name.split('_')[0]
    userid = int(name.split('_')[1])
    current_time = datetime.now()
    today_date = date.today()

    existing_record = Attendance.query.filter_by(roll=userid, date=today_date).first()
    if not existing_record:
        new_attendance = Attendance(name=username, roll=userid, time=current_time, date=today_date)
        db.session.add(new_attendance)
        db.session.commit()

## A function to get names and rol numbers of all users
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l

## A function to delete a user folder
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser + '/' + i)
    os.rmdir(duser)

################## ROUTING FUNCTIONS #########################

# Our main page
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

## List users page
@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)

## Delete functionality
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/' + duser)

    ## if all the face are deleted, delete the trained file...
    if os.listdir('static/faces/') == []:
        os.remove('static/face_recognition_model.pkl')

    try:
        train_model()
    except:
        pass

    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# Our main Face Recognition functionality.
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['POST'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    data = request.get_json()
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if len(extract_faces(frame)) > 0:
        (x, y, w, h) = extract_faces(frame)[0]
        face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
        identified_person = identify_face(face.reshape(1, -1))[0]
        add_attendance(identified_person)
        return jsonify({'message': f'Attendance marked for {identified_person}'})
    else:
        return jsonify({'message': 'No face detected'})

# A function to add a new user.
# This function will run when we add a new user.
@app.route('/add', methods=['POST'])
def add():
    data = request.get_json()
    newusername = data['newusername']
    newuserid = data['newuserid']
    images_data = data['images']

    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    for i, image_data in enumerate(images_data):
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            name = f'{newusername}_{i}.jpg'
            cv2.imwrite(os.path.join(userimagefolder, name), frame[y:y + h, x:x + w])

    print('Training Model')
    train_model()
    return jsonify({'message': f'User {newusername} added successfully'})

if __name__ == '__main__':
    # Create a db and table
    with app.app_context():
        db.create_all()
    app.run(debug=True)