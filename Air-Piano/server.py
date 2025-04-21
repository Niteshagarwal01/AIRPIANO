import os
from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import threading
import base64
import json
import time
import pygame.midi
from cvzone.HandTrackingModule import HandDetector

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'airpiano-secret!'
socketio = SocketIO(app, cors_allowed_origins='*')

# Initialize pygame MIDI
pygame.midi.init()
player = pygame.midi.Output(0)
player.set_instrument(0)  # Acoustic Grand Piano

# Initialize Hand Detector
cap = None  # Will be initialized when needed
detector = HandDetector(detectionCon=0.8)

# Chord Mapping - Same as in hand_dscale.py
chords = {
    "left": {
        "thumb": [62, 66, 69],   # D Major (D, F#, A)
        "index": [64, 67, 71],   # E Minor (E, G, B)
        "middle": [67, 71, 74],  # G Major (G, B, D)
        "ring": [69, 73, 76],    # A Major (A, C#, E)
        "pinky": [62, 65, 69]    # D Minor (D, F, A)
    },
    "right": {
        "thumb": [69, 73, 76],   # A Major (A, C#, E)
        "index": [71, 74, 78],   # B Minor (B, D, F#)
        "middle": [66, 69, 73],  # F# Minor (F#, A, C#)
        "ring": [67, 71, 74],    # G Major (G, B, D)
        "pinky": [62, 66, 69, 73]  # D Major/7th (D, F#, A, C#)
    }
}

# Combination Chords (from Chord_mapping.txt)
combination_chords = {
    "left": {
        "thumb+index": [61, 64, 67],      # C# Diminished (C#, E, G)
        "index+middle": [64, 68, 71, 74], # E7 (E, G#, B, D)
        "middle+ring": [67, 70, 74],      # G Minor (G, Bb, D)
        "ring+pinky": [69, 73, 76, 79],   # A7 (A, C#, E, G)
        "thumb+pinky": [62, 66, 69, 72]   # D9 (D, F#, A, C)
    },
    "right": {
        "thumb+index": [69, 74, 76],      # A sus4 (A, D, E)
        "index+middle": [66, 70, 73, 76], # F#7 (F#, A#, C#, E)
        "middle+ring": [67, 71, 74, 77],  # G7 (G, B, D, F)
        "ring+pinky": [71, 76, 78],       # B sus4 (B, E, F#)
        "thumb+pinky": [62, 69, 73, 78]   # D13 (D, A, C#, F#)
    }
}

# Sustain Time (in seconds)
SUSTAIN_TIME = 2.0

# Track Previous States to Stop Chords
prev_states = {hand: {finger: 0 for finger in chords[hand]} for hand in chords}
prev_combo_states = {hand: {combo: 0 for combo in combination_chords[hand]} for hand in combination_chords}

# Mapping of finger indices to names
finger_names = ["thumb", "index", "middle", "ring", "pinky"]

# Flag to control the camera loop
running = False
thread = None

# Function to Play a Chord
def play_chord(chord_notes):
    for note in chord_notes:
        player.note_on(note, 127)

# Function to Stop a Chord After a Delay
def stop_chord_after_delay(chord_notes):
    time.sleep(SUSTAIN_TIME)
    for note in chord_notes:
        player.note_off(note, 127)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    global running
    running = False

@socketio.on('start_tracking')
def handle_start_tracking():
    global running, cap, thread
    
    if not running:
        cap = cv2.VideoCapture(0)
        running = True
        thread = threading.Thread(target=camera_loop)
        thread.daemon = True
        thread.start()

@socketio.on('stop_tracking')
def handle_stop_tracking():
    global running
    running = False
    if cap is not None:
        cap.release()

def camera_loop():
    global running, prev_states, prev_combo_states
    
    while running:
        success, img = cap.read()
        if not success:
            print("Camera not capturing frames")
            socketio.emit('error', {'message': 'Camera not capturing frames'})
            time.sleep(0.1)
            continue

        # Flip image horizontally for a more natural view
        img = cv2.flip(img, 1)
        
        # Find hands
        hands, img = detector.findHands(img, draw=True)
        
        # Data structure to send to frontend
        hand_data = []
        active_chords = []

        if hands:
            for hand in hands:
                hand_type = "left" if hand["type"] == "Left" else "right"
                fingers = detector.fingersUp(hand)
                
                # Get landmarks for visualization
                landmarks = hand["lmList"]  # List of 21 landmarks [id, x, y, z]
                
                # Add hand data for visualization
                hand_data.append({
                    'type': hand_type,
                    'landmarks': landmarks,
                    'fingers': fingers
                })
                
                # Process single finger chords
                for i, finger_status in enumerate(fingers):
                    finger = finger_names[i]
                    if finger in chords[hand_type]:
                        if finger_status == 1 and prev_states[hand_type][finger] == 0:
                            chord_notes = chords[hand_type][finger]
                            play_chord(chord_notes)
                            chord_name = get_chord_name(hand_type, finger, "single")
                            active_chords.append({
                                'hand': hand_type, 
                                'finger': finger, 
                                'type': 'single',
                                'name': chord_name
                            })
                        elif finger_status == 0 and prev_states[hand_type][finger] == 1:
                            chord_notes = chords[hand_type][finger]
                            threading.Thread(
                                target=stop_chord_after_delay, 
                                args=(chord_notes,), 
                                daemon=True
                            ).start()
                        prev_states[hand_type][finger] = finger_status
                
                # Process combination chords
                # Check for thumb+index
                if fingers[0] == 1 and fingers[1] == 1 and all(f == 0 for f in fingers[2:]):
                    combo = "thumb+index"
                    if prev_combo_states[hand_type][combo] == 0:
                        chord_notes = combination_chords[hand_type][combo]
                        play_chord(chord_notes)
                        chord_name = get_chord_name(hand_type, combo, "combo")
                        active_chords.append({
                            'hand': hand_type, 
                            'finger': combo, 
                            'type': 'combo',
                            'name': chord_name
                        })
                    prev_combo_states[hand_type][combo] = 1
                else:
                    combo = "thumb+index"
                    if prev_combo_states[hand_type][combo] == 1:
                        chord_notes = combination_chords[hand_type][combo]
                        threading.Thread(
                            target=stop_chord_after_delay, 
                            args=(chord_notes,), 
                            daemon=True
                        ).start()
                    prev_combo_states[hand_type][combo] = 0
                
                # Similar logic for other combinations
                # index+middle
                if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 and all(f == 0 for f in fingers[3:]):
                    combo = "index+middle"
                    if prev_combo_states[hand_type][combo] == 0:
                        chord_notes = combination_chords[hand_type][combo]
                        play_chord(chord_notes)
                        chord_name = get_chord_name(hand_type, combo, "combo")
                        active_chords.append({
                            'hand': hand_type, 
                            'finger': combo, 
                            'type': 'combo',
                            'name': chord_name
                        })
                    prev_combo_states[hand_type][combo] = 1
                else:
                    combo = "index+middle"
                    if prev_combo_states[hand_type][combo] == 1:
                        chord_notes = combination_chords[hand_type][combo]
                        threading.Thread(
                            target=stop_chord_after_delay, 
                            args=(chord_notes,), 
                            daemon=True
                        ).start()
                    prev_combo_states[hand_type][combo] = 0
                
                # Add other combinations as needed
        
        # Convert the image to JPEG
        _, buffer = cv2.imencode('.jpg', img)
        img_data = base64.b64encode(buffer).decode('utf-8')
        
        # Send data to clients
        socketio.emit('camera_frame', {
            'image': img_data,
            'hands': hand_data,
            'active_chords': active_chords
        })
        
        # Small delay to reduce CPU usage
        time.sleep(0.03)

def get_chord_name(hand_type, finger, chord_type):
    """Get the human-readable chord name"""
    chord_mappings = {
        "left": {
            "single": {
                "thumb": "D Major",
                "index": "E Minor",
                "middle": "G Major",
                "ring": "A Major",
                "pinky": "D Minor"
            },
            "combo": {
                "thumb+index": "C# Diminished",
                "index+middle": "E7",
                "middle+ring": "G Minor",
                "ring+pinky": "A7",
                "thumb+pinky": "D9"
            }
        },
        "right": {
            "single": {
                "thumb": "A Major",
                "index": "B Minor",
                "middle": "F# Minor",
                "ring": "G Major",
                "pinky": "D Major/7th"
            },
            "combo": {
                "thumb+index": "A sus4",
                "index+middle": "F#7",
                "middle+ring": "G7",
                "ring+pinky": "B sus4",
                "thumb+pinky": "D13"
            }
        }
    }
    
    return chord_mappings[hand_type][chord_type].get(finger, "Unknown Chord")

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)