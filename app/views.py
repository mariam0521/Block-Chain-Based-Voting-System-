import datetime
import json
import os
import requests
from flask import render_template, redirect, request, flash, session, Response, jsonify
from werkzeug.utils import secure_filename
from app import app
import face_recognition
import numpy as np
import cv2
import time

# Configuration for Aadhar-based voter registration
AADHAR_VOTERS_FILE = 'aadhar_voters.json'
UPLOAD_FOLDER = 'face_captures'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# The node with which our application interacts
CONNECTED_SERVICE_ADDRESS = "http://127.0.0.1:8000"
POLITICAL_PARTIES = ["Democratic Party","Republican Party","Socialist party"]

# Global variables
vote_check = []
posts = []
current_face_capture = None
face_detection_in_progress = False


def fetch_posts():
    """
    Function to fetch the chain from a blockchain node, parse the
    data and store it locally.
    """
    get_chain_address = "{}/chain".format(CONNECTED_SERVICE_ADDRESS)
    response = requests.get(get_chain_address)
    if response.status_code == 200:
        content = []
        vote_count = []
        chain = json.loads(response.content)
        for block in chain["chain"]:
            for tx in block["transactions"]:
                tx["index"] = block["index"]
                tx["hash"] = block["previous_hash"]
                content.append(tx)


        global posts
        posts = sorted(content, key=lambda k: k['timestamp'],
                       reverse=True)


@app.route('/')
def index():
    fetch_posts()

    vote_gain = []
    VOTER_IDS = []
    for post in posts:
        vote_gain.append(post["party"])

    return render_template('index.html',
                           title='VOTING SYSTEM',
                           posts=posts,
                           vote_gain=vote_gain,
                           node_address=CONNECTED_SERVICE_ADDRESS,
                           readable_time=timestamp_to_string,
                           political_parties=POLITICAL_PARTIES,
                           voter_ids=VOTER_IDS)


def load_aadhar_voters():
    """Load registered Aadhar voters from file."""
    try:
        with open(AADHAR_VOTERS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_aadhar_voters(voters):
    """Save Aadhar voters to file."""
    with open(AADHAR_VOTERS_FILE, 'w') as f:
        json.dump(voters, f)

def detect_and_extract_face(frame):
    """
    Detect and extract face from a frame using Haar Cascade
    Returns the extracted face or None
    """
    # Load pre-trained face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        # Get the first detected face
        (x, y, w, h) = faces[0]
        
        # Extract face
        face_img = frame[y:y+h, x:x+w]
        
        return face_img, (x, y, w, h)
    
    return None, None

def generate_frames(mode='registration'):
    """
    Generate frames for live face detection
    mode can be 'registration' or 'verification'
    """
    global current_face_capture, face_detection_in_progress
    
    # Open webcam
    video_capture = cv2.VideoCapture(0)
    
    # Counters and flags
    capture_count = 0
    max_captures = 10  # Increased from 5 to 10 for more reliable capture
    
    start_time = time.time()
    timeout = 10  # 10 seconds timeout
    
    while True:
        # Check timeout
        if time.time() - start_time > timeout:
            face_detection_in_progress = False
            video_capture.release()
            break
        
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        
        if not ret:
            break
        
        # Detect face
        face_detection_result = detect_and_extract_face(frame)
        
        # Check if a face was detected
        if face_detection_result[0] is not None:
            face, (x, y, w, h) = face_detection_result
            
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Increment capture count
            capture_count += 1
            
            # If enough captures, save the last detected face
            if capture_count >= max_captures:
                current_face_capture = face
                face_detection_in_progress = False
                video_capture.release()
                break
        
        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    # Release resources
    video_capture.release()

@app.route('/video_feed')
def video_feed():
    """Route to stream video feed for face detection"""
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['GET', 'POST'])
def register():
    global current_face_capture, face_detection_in_progress
    
    if request.method == 'POST':
        # Reset face capture
        current_face_capture = None
        face_detection_in_progress = True
        
        # Get Aadhar number
        aadhar_number = request.form['aadhar_number']
        
        # Verify Aadhar number format
        if not aadhar_number.isdigit() or len(aadhar_number) != 12:
            return jsonify({
                'status': 'error', 
                'message': 'Invalid Aadhar Number. Must be 12 digits.'
            }), 400
        
        # Check if Aadhar number already exists
        aadhar_voters = load_aadhar_voters()
        if aadhar_number in aadhar_voters:
            return jsonify({
                'status': 'already_registered', 
                'message': f'Aadhar number {aadhar_number} is already registered!'
            }), 409
        
        # Handle face capture
        if 'face_capture' not in request.files:
            return jsonify({
                'status': 'error', 
                'message': 'No face capture found'
            }), 400
        
        face_file = request.files['face_capture']
        
        # Save captured face
        filename = f"{aadhar_number}_face.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        face_file.save(filepath)
        
        # Save voter details
        aadhar_voters[aadhar_number] = {
            'face_image': filepath,
            'registered_at': datetime.datetime.now().isoformat()
        }
        save_aadhar_voters(aadhar_voters)
        
        return jsonify({
            'status': 'success', 
            'message': 'Successfully registered!'
        }), 201
    
    return render_template('register.html', title='Voter Registration')

@app.route('/submit', methods=['POST'])
def submit_textarea():
    """Endpoint to create a new transaction with live face verification."""
    global current_face_capture, face_detection_in_progress
    
    try:
        # Get form data
        aadhar_number = request.form["aadhar_number"]
        party = request.form["party"]
        
        # Verify Aadhar number
        aadhar_voters = load_aadhar_voters()
        if aadhar_number not in aadhar_voters:
            flash('Aadhar number not registered!', 'error')
            return redirect('/')
        
        # Check if already voted
        if aadhar_number in vote_check:
            flash('Aadhar number ('+aadhar_number+') has already voted!', 'error')
            return redirect('/')
        
        # Reset face capture and start detection
        current_face_capture = None
        face_detection_in_progress = True
        
        # Generate frames to capture face
        for _ in generate_frames(mode='verification'):
            if current_face_capture is not None:
                break
        
        # Check if face was captured
        if current_face_capture is None:
            flash('Face verification failed. No face detected.', 'error')
            return redirect('/')
        
        # Compare faces
        try:
            registered_face = cv2.imread(aadhar_voters[aadhar_number]['face_image'])
            
            # Check if registered face image exists and can be read
            if registered_face is None:
                flash('Registered face image not found!', 'error')
                return redirect('/')
            
            # Convert to RGB for face recognition
            registered_face_rgb = cv2.cvtColor(registered_face, cv2.COLOR_BGR2RGB)
            current_face_rgb = cv2.cvtColor(current_face_capture, cv2.COLOR_BGR2RGB)
            
            # Encode faces
            registered_encodings = face_recognition.face_encodings(registered_face_rgb)
            current_encodings = face_recognition.face_encodings(current_face_rgb)
            
            # Check if face encodings were successfully generated
            if not registered_encodings or not current_encodings:
                flash('Face encoding failed!', 'error')
                return redirect('/')
            
            # Compare face encodings
            face_matches = face_recognition.compare_faces(
                [registered_encodings[0]], 
                current_encodings[0]
            )
            
            if not face_matches[0]:
                flash('Face verification failed!', 'error')
                return redirect('/')
        
        except Exception as e:
            flash(f'Face verification error: {str(e)}', 'error')
            return redirect('/')
        
        # Prepare vote transaction
        post_object = {
            'voter_id': aadhar_number,
            'party': party,
        }
        
        # Submit transaction
        new_tx_address = "{}/new_transaction".format(CONNECTED_SERVICE_ADDRESS)
        response = requests.post(new_tx_address,
                      json=post_object,
                      headers={'Content-type': 'application/json'})
        
        # Check transaction submission
        if response.status_code != 201:
            flash('Failed to submit vote!', 'error')
            return redirect('/')
        
        # Mark as voted
        vote_check.append(aadhar_number)
        
        flash('Voted to '+party+' successfully!', 'success')
        return redirect('/')
    
    except Exception as e:
        flash(f'Unexpected error: {str(e)}', 'error')
        return redirect('/')


def timestamp_to_string(epoch_time):
    return datetime.datetime.fromtimestamp(epoch_time).strftime('%Y-%m-%d %H:%M')
