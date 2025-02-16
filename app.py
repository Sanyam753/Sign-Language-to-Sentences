import os
import cv2
import time
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory, jsonify
from matplotlib import pyplot as plt
import tensorflow as tf
# TensorFlow and Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # folder to save uploaded videos

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Set up TensorBoard callback directory (optional)
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# MediaPipe models and drawing utilities
mp_holistic = mp.solutions.holistic      # Holistic model
mp_face_mesh = mp.solutions.face_mesh    # Face Mesh model
mp_drawing = mp.solutions.drawing_utils   # Drawing utilities

# ------------------- Helper Functions ------------------- #

def mediapipe_detection(image, model):
    """Converts image to RGB, processes it with MediaPipe, then converts back to BGR."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    """Draws styled landmarks with custom colors and thickness."""
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, 
        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    )
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )

def extract_keypoints(results):
    """Extracts keypoints from MediaPipe results into a single flattened array."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions, input_frame, colors):
    """Visualizes the probability of each action on the frame."""
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# ------------------- Model Setup ------------------- #

actions = np.array(['hello', 'thanks', 'iloveyou'])

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.load_weights(r'C:\Users\anshu\OneDrive\Documents\Web-Dev-Prac\Sign-Language-to-Sentences\model\action.h5')

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

# ------------------- Global Variables ------------------- #
sequence = []
sentence = []
predictions = []
threshold = 0.5

# ------------------- Video Streaming Generator ------------------- #

def generate_frames(source=0):
    """
    Video streaming generator function.
    Parameter 'source' can be 0 for live capture or a filepath for an uploaded video.
    """
    global sequence, sentence, predictions
    cap = cv2.VideoCapture(source)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                
                if len(predictions) >= 10 and len(set(predictions[-10:])) == 1:
                    if res[np.argmax(res)] > threshold:
                        if not sentence or actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                if len(sentence) > 5:
                    sentence = sentence[-5:]

                image = prob_viz(res, actions, image, colors)
            
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', image)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

# ------------------- Flask Routes ------------------- #

@app.route('/')
def home():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/index')
def index_page():
    """Render alternate homepage."""
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/starter-page')
def starter():
    return render_template('starter-page.html')

@app.route('/portfolio-details')
def portfolio_details():
    return render_template('portfolio-details.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route for live feed."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/files')
def files_page():
    """Render the files page with both options."""
    return render_template('files.html')

@app.route('/live_video_feed')
def live_video_feed():
    """Route to stream live video feed."""
    global sequence, sentence, predictions
    sequence, sentence, predictions = [], [], []
    return render_template('live.html')

@app.route('/live_stream')
def live_stream():
    """Stream live video feed frames."""
    return Response(generate_frames(source=0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/uploaded_video_feed/<filename>')
def uploaded_video_feed(filename):
    """Route to stream the uploaded video."""
    global sequence, sentence, predictions
    sequence, sentence, predictions = [], [], []
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return render_template('uploaded.html', filename=filename)

@app.route('/uploaded_stream/<filename>')
def uploaded_stream(filename):
    """Stream the uploaded video frames."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(generate_frames(source=filepath),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    """Route for uploading a video file."""
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)
        file = request.files['video']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            return redirect(url_for('upload_video', filename=file.filename))
    filename = request.args.get('filename')
    return render_template('upload.html', filename=filename)

@app.route('/detection_text')
def detection_text():
    """Return the current detection text as JSON."""
    global sentence
    current_text = sentence[-1] if sentence else ""
    return jsonify({"text": current_text})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ------------------- Main ------------------- #

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
