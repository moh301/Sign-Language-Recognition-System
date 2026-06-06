#python app.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import json
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import pickle
import tensorflow as tf
import time
from collections import deque
from gtts import gTTS

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Authentication
ACCOUNT_SETTINGS_FILE = 'account_settings.json'
DEFAULT_ACCOUNT_SETTINGS = {
    'display_name': '',
    'password': 'moaz-JBL'
}


def load_account_settings():
    if os.path.exists(ACCOUNT_SETTINGS_FILE):
        try:
            with open(ACCOUNT_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    'display_name': str(data.get('display_name', '')).strip(),
                    'password': str(data.get('password', DEFAULT_ACCOUNT_SETTINGS['password']))
                }
        except Exception as e:
            print(f"⚠️ Error loading account settings: {e}")

    return DEFAULT_ACCOUNT_SETTINGS.copy()


def save_account_settings(settings):
    with open(ACCOUNT_SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2)


account_settings = load_account_settings()

# ===== CONFIG =====
CONF_THRESHOLD = 0.80
SYSTEM_GESTURES = {"NO_HAND", "NO_SIGN", "WAITING", "LOW_CONFIDENCE"}

prediction_buffer = deque(maxlen=7)
last_emitted = {"label": None, "time": 0}

# ===== MediaPipe =====
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    USE_MEDIAPIPE = True
    print("✅ MediaPipe initialized")
except Exception as e:
    print(f"⚠️ MediaPipe error: {e}")
    USE_MEDIAPIPE = False
    hands_detector = None

# ===== Storage =====
recognized_signs = []

# ===== Load Model =====
print("Loading gesture recognition model...")
model = None
scaler = None
label_map = {}

try:
    model = tf.keras.models.load_model('static/model/gesture_model.keras')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

try:
    with open('static/model/scaler_v2.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Scaler loaded successfully")
except Exception as e:
    print(f"❌ Error loading scaler: {e}")

try:
    with open('static/model/label_map_v2.json', 'r') as f:
        label_map = json.load(f)
    print("✅ Label map loaded successfully")
except Exception as e:
    print(f"❌ Error loading label map: {e}")


# ===== Extract Landmarks =====
def extract_landmarks_from_image(image_data):
    try:
        if isinstance(image_data, Image.Image):
            frame = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
        else:
            frame = image_data
        
        if USE_MEDIAPIPE and hands_detector:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                features = []
                for lm in hand_landmarks.landmark:
                    features.extend([float(lm.x), float(lm.y), float(lm.z)])
                
                if len(features) == 63:
                    return np.array(features, dtype=np.float32).reshape(1, 63)
        
        return np.zeros((1, 63), dtype=np.float32)
    
    except Exception as e:
        print(f"Error extracting landmarks: {e}")
        return np.zeros((1, 63), dtype=np.float32)


# ===== Routes =====
@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    global account_settings

    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        saved_name = account_settings.get('display_name', '').strip()
        
        valid_name = username and (not saved_name or username == saved_name)

        if valid_name and password == account_settings['password']:
            if not saved_name:
                account_settings['display_name'] = username
                save_account_settings(account_settings)

            session['username'] = username
            return jsonify({'success': True})
        else:
            return jsonify({'success': False})
    
    return render_template('login.html')


@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('welcome.html', username=session['username'])


@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])


@app.route('/account')
def account():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('account.html', username=session['username'])


@app.route('/api/account/update-name', methods=['POST'])
def update_account_name():
    global account_settings

    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401

    data = request.get_json() or {}
    new_name = data.get('new_name', '').strip()

    if len(new_name) < 2:
        return jsonify({'success': False, 'message': 'Name must be at least 2 characters.'}), 400

    account_settings['display_name'] = new_name
    save_account_settings(account_settings)
    session['username'] = new_name

    return jsonify({'success': True, 'username': new_name})


@app.route('/api/account/update-password', methods=['POST'])
def update_account_password():
    global account_settings

    if 'username' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401

    data = request.get_json() or {}
    current_password = data.get('current_password', '')
    new_password = data.get('new_password', '')
    confirm_password = data.get('confirm_password', '')

    if current_password != account_settings['password']:
        return jsonify({'success': False, 'message': 'Current password is incorrect.'}), 400

    if len(new_password) < 4:
        return jsonify({'success': False, 'message': 'New password must be at least 4 characters.'}), 400

    if new_password != confirm_password:
        return jsonify({'success': False, 'message': 'New password and confirmation do not match.'}), 400

    account_settings['password'] = new_password
    save_account_settings(account_settings)

    return jsonify({'success': True})


# ===== Detection API =====
@app.route('/api/detect', methods=['POST'])
def detect_gesture():
    try:
        if not model:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        image_data = data.get('image', '')
        
        if not image_data:
            return jsonify({'success': False, 'error': 'No image'}), 400
        
        # decode image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # ===== extract features =====
        features = extract_landmarks_from_image(image)

        # ===== NO HAND =====
        if np.all(features == 0):
            return jsonify({
                'success': True,
                'gesture': "NO_HAND",
                'confidence': 1.0,
                'class': -1
            })
        
        # normalize
        if scaler:
            features = scaler.transform(features)
        
        # ===== predict =====
        prediction = model.predict(features, verbose=0)
        confidence = float(np.max(prediction))
        predicted_class = int(np.argmax(prediction))

        # ===== confidence filter =====
        if confidence < CONF_THRESHOLD:
            return jsonify({
                'success': True,
                'gesture': "LOW_CONFIDENCE",
                'confidence': confidence,
                'class': predicted_class
            })

        # ===== smoothing =====
        prediction_buffer.append(predicted_class)

        if len(prediction_buffer) < 7:
            final_class = predicted_class
        else:
            final_class = max(set(prediction_buffer), key=prediction_buffer.count)

        gesture_name = label_map.get(str(final_class), "NO_SIGN")

        # ===== ignore NO_HAND from model =====
        if gesture_name == "NO_HAND":
            return jsonify({
                'success': True,
                'gesture': "NO_HAND",
                'confidence': confidence,
                'class': final_class
            })

        # If the detected sign did not change, keep waiting until a new sign appears.
        if gesture_name == last_emitted["label"]:
            return jsonify({
                'success': True,
                'gesture': "WAITING",
                'confidence': confidence,
                'class': final_class
            })

        # update
        last_emitted["label"] = gesture_name
        last_emitted["time"] = time.time()

        return jsonify({
            'success': True,
            'gesture': gesture_name,
            'confidence': confidence,
            'class': final_class
        })

    except Exception as e:
        print(f"Error in detect: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ===== History APIs =====
@app.route('/api/save-sign', methods=['POST'])
def save_sign():
    try:
        data = request.get_json()
        sign = data.get('sign', '')
        confidence = data.get('confidence', 0)
        
        if sign and sign not in SYSTEM_GESTURES:
            recognized_signs.append({
                'sign': sign,
                'confidence': round(confidence * 100, 2)
            })

            if len(recognized_signs) > 100:
                recognized_signs.pop(0)

            return jsonify({'success': True})
        else:
            return jsonify({'success': False})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/get-signs', methods=['GET'])
def get_signs():
    return jsonify({'signs': recognized_signs})


@app.route('/api/clear-signs', methods=['POST'])
def clear_signs():
    global recognized_signs
    recognized_signs = []
    return jsonify({'success': True})


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ===== Creator Mode Routes =====
VIDEOS_DIR = 'static/videos'
CREATOR_VIDEOS_FILE = 'creator_videos.json'

# Ensure videos directory exists
if not os.path.exists(VIDEOS_DIR):
    os.makedirs(VIDEOS_DIR)


def load_creator_videos():
    """Load creator videos metadata from JSON"""
    if os.path.exists(CREATOR_VIDEOS_FILE):
        try:
            with open(CREATOR_VIDEOS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading creator videos: {e}")
    return []


def save_creator_videos(videos):
    """Save creator videos metadata to JSON"""
    try:
        with open(CREATOR_VIDEOS_FILE, 'w', encoding='utf-8') as f:
            json.dump(videos, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Error saving creator videos: {e}")


@app.route('/creator')
def creator():
    """Creator Mode page"""
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('creator.html', username=session['username'])


@app.route('/api/creator/save-video', methods=['POST'])
def save_creator_video():
    """Save recorded video"""
    try:
        if 'username' not in session:
            return jsonify({'success': False, 'message': 'Unauthorized'}), 401

        if 'video' not in request.files:
            return jsonify({'success': False, 'message': 'No video file'}), 400

        video_file = request.files['video']
        transcript = request.form.get('transcript', '')
        language = request.form.get('language', 'en').strip().lower()

        if language not in {'en', 'ar', 'fr', 'es'}:
            language = 'en'

        if video_file.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'}), 400

        # Generate unique filename
        filename = f"video_{session['username']}_{int(time.time())}.webm"
        filepath = os.path.join(VIDEOS_DIR, filename)

        # Save video file
        video_file.save(filepath)

        # Save metadata
        videos = load_creator_videos()
        videos.append({
            'filename': filename,
            'transcript': transcript,
            'language': language,
            'created_by': session['username'],
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'filesize': os.path.getsize(filepath)
        })
        save_creator_videos(videos)

        return jsonify({'success': True, 'filename': filename})

    except Exception as e:
        print(f"❌ Error saving creator video: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/creator/tts', methods=['POST'])
def creator_tts():
    """Generate TTS audio for creator mode using four supported languages only."""
    try:
        if 'username' not in session:
            return jsonify({'success': False, 'message': 'Unauthorized'}), 401

        data = request.get_json() or {}
        text = str(data.get('text', '')).strip()
        language = str(data.get('language', 'en')).strip().lower()

        if not text:
            return jsonify({'success': False, 'message': 'No text provided'}), 400

        if len(text) > 250:
            text = text[:250]

        language_map = {
            'en': 'en',
            'ar': 'ar',
            'fr': 'fr',
            'es': 'es'
        }
        tts_lang = language_map.get(language, 'en')

        audio_bytes = BytesIO()
        tts = gTTS(text=text, lang=tts_lang)
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)

        return app.response_class(audio_bytes.getvalue(), mimetype='audio/mpeg')

    except Exception as e:
        print(f"❌ Error generating creator TTS: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/creator/get-videos', methods=['GET'])
def get_creator_videos():
    """Get all creator videos"""
    try:
        if 'username' not in session:
            return jsonify({'success': False, 'message': 'Unauthorized'}), 401

        videos = load_creator_videos()
        # Filter videos by current user
        user_videos = [v for v in videos if v.get('created_by') == session['username']]
        
        return jsonify({'videos': user_videos})

    except Exception as e:
        print(f"❌ Error getting creator videos: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/creator/get-video/<filename>', methods=['GET'])
def get_creator_video(filename):
    """Get specific creator video metadata"""
    try:
        if 'username' not in session:
            return jsonify({'success': False, 'message': 'Unauthorized'}), 401

        videos = load_creator_videos()
        video = next((v for v in videos if v['filename'] == filename), None)

        if not video:
            return jsonify({'success': False, 'message': 'Video not found'}), 404

        # Security: ensure user owns this video
        if video.get('created_by') != session['username']:
            return jsonify({'success': False, 'message': 'Unauthorized'}), 401

        return jsonify({
            'success': True,
            'filename': video['filename'],
            'transcript': video['transcript'],
            'created_at': video['created_at']
        })

    except Exception as e:
        print(f"❌ Error getting creator video: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/creator/delete-video/<filename>', methods=['DELETE'])
def delete_creator_video(filename):
    """Delete creator video"""
    try:
        if 'username' not in session:
            return jsonify({'success': False, 'message': 'Unauthorized'}), 401

        # Security: check if file belongs to user
        videos = load_creator_videos()
        video = next((v for v in videos if v['filename'] == filename), None)

        if not video:
            return jsonify({'success': False, 'message': 'Video not found'}), 404

        if video.get('created_by') != session['username']:
            return jsonify({'success': False, 'message': 'Unauthorized'}), 401

        # Delete file
        filepath = os.path.join(VIDEOS_DIR, filename)
        if os.path.exists(filepath):
            os.remove(filepath)

        # Update metadata
        videos = [v for v in videos if v['filename'] != filename]
        save_creator_videos(videos)

        return jsonify({'success': True})

    except Exception as e:
        print(f"❌ Error deleting creator video: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)