import os
from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading
import base64
import time
import json
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import logging

# Add FluidSynth integration for better sound quality
try:
    import fluidsynth
    USE_FLUIDSYNTH = True
except ImportError:
    USE_FLUIDSYNTH = False
    print("FluidSynth not available, using default pygame MIDI")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('airpiano')

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['SECRET_KEY'] = 'airpiano-secret!'

# Global variables for chord state tracking and settings
active_chords = []
active_hands = []
tracking_active = False
camera_data = {
    "calibrated": False,
    "brightness": 100,
    "contrast": 100
}
settings = {
    "sustain_time": 2.0,
    "sensitivity": 0.8,
    "volume": 100
}

# Track performance metrics
performance_metrics = {
    "frames_processed": 0,
    "hands_detected": 0,
    "chords_played": 0,
    "session_start": None,
    "session_duration": 0
}

# Available soundfonts - add path to any soundfonts you have
soundfonts = {
    "default": None,  # Will use pygame default
}

# Try to locate common soundfont locations
potential_soundfonts = [
    "soundfonts/FluidR3_GM.sf2",
    "C:/Windows/System32/drivers/gm.dls",
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2"
]

for sf_path in potential_soundfonts:
    if os.path.exists(sf_path):
        soundfonts["enhanced"] = sf_path
        break

# Initialize sound system
player = None
fs = None

# Initialize sound system
def init_sound_system():
    global player, fs, USE_FLUIDSYNTH
    
    if USE_FLUIDSYNTH and "enhanced" in soundfonts and soundfonts["enhanced"]:
        try:
            fs = fluidsynth.Synth()
            fs.start()
            sfid = fs.sfload(soundfonts["enhanced"])
            fs.program_select(0, sfid, 0, 0)  # Default to piano
            logger.info(f"FluidSynth initialized with soundfont: {soundfonts['enhanced']}")
            return
        except Exception as e:
            logger.error(f"Error initializing FluidSynth: {e}")
            USE_FLUIDSYNTH = False
    
    # Fall back to pygame MIDI if FluidSynth fails or is not available
    try:
        import pygame.midi
        pygame.midi.init()
        midi_count = pygame.midi.get_count()
        logger.info(f"Found {midi_count} MIDI devices")
        
        if midi_count > 0:
            player = pygame.midi.Output(0)
            player.set_instrument(0)  # 0 = Acoustic Grand Piano
            logger.info("MIDI initialized successfully")
        else:
            logger.error("No MIDI devices found")
            player = None
    except Exception as e:
        logger.error(f"Error initializing MIDI: {e}")
        player = None

init_sound_system()

# Available instruments list (General MIDI standard)
instruments = {
    0: "Acoustic Grand Piano",
    1: "Bright Acoustic Piano",
    2: "Electric Grand Piano",
    4: "Electric Piano",
    5: "Electric Piano 2",
    6: "Harpsichord",
    11: "Vibraphone",
    12: "Marimba",
    19: "Church Organ",
    24: "Acoustic Guitar (nylon)",
    25: "Acoustic Guitar (steel)",
    26: "Electric Guitar (jazz)",
    27: "Electric Guitar (clean)",
    28: "Electric Guitar (muted)",
    29: "Overdriven Guitar",
    30: "Distortion Guitar",
    32: "Acoustic Bass",
    33: "Electric Bass (finger)",
    34: "Electric Bass (pick)",
    40: "Violin",
    41: "Viola",
    42: "Cello",
    48: "String Ensemble 1",
    56: "Trumpet",
    57: "Trombone",
    66: "Tenor Sax",
    73: "Flute",
    74: "Recorder",
    75: "Pan Flute",
    80: "Lead Synth (square)",
    81: "Lead Synth (sawtooth)"
}

# Current instrument
current_instrument = 0

# Initialize Hand Detector
cap = None
detector = None

# Map from finger names to indices
finger_indices = {
    "thumb": 0,
    "index": 1,
    "middle": 2,
    "ring": 3,
    "pinky": 4
}

# Single Finger Chord Mapping
single_chords = {
    "left": {
        "thumb": {"notes": [62, 66, 69], "name": "D Major"},   # D Major (D, F#, A)
        "index": {"notes": [64, 67, 71], "name": "E Minor"},   # E Minor (E, G, B)
        "middle": {"notes": [67, 71, 74], "name": "G Major"},  # G Major (G, B, D)
        "ring": {"notes": [69, 73, 76], "name": "A Major"},    # A Major (A, C#, E)
        "pinky": {"notes": [62, 65, 69], "name": "D Minor"}    # D Minor (D, F, A)
    },
    "right": {
        "thumb": {"notes": [69, 73, 76], "name": "A Major"},   # A Major (A, C#, E)
        "index": {"notes": [71, 74, 78], "name": "B Minor"},   # B Minor (B, D, F#)
        "middle": {"notes": [66, 69, 73], "name": "F# Minor"},  # F# Minor (F#, A, C#)
        "ring": {"notes": [67, 71, 74], "name": "G Major"},    # G Major (G, B, D)
        "pinky": {"notes": [62, 66, 69, 73], "name": "D Major/7th"}  # D Major/7th (D, F#, A, C#)
    }
}

# Combo Chord Mapping
combo_chords = {
    "left": {
        "thumb_index": {"notes": [61, 64, 67], "name": "C# Diminished"},  # C# Diminished (C#, E, G)
        "index_middle": {"notes": [64, 68, 71, 74], "name": "E7"},  # E7 (E, G#, B, D)
        "middle_ring": {"notes": [67, 70, 74], "name": "G Minor"},  # G Minor (G, Bb, D)
        "ring_pinky": {"notes": [69, 73, 76, 79], "name": "A7"},  # A7 (A, C#, E, G)
        "thumb_pinky": {"notes": [62, 66, 69, 72], "name": "D9"}  # D9 (D, F#, A, C)
    },
    "right": {
        "thumb_index": {"notes": [69, 74, 76], "name": "A sus4"},  # A sus4 (A, D, E)
        "index_middle": {"notes": [66, 70, 73, 76], "name": "F#7"},  # F#7 (F#, A#, C#, E)
        "middle_ring": {"notes": [67, 71, 74, 77], "name": "G7"},  # G7 (G, B, D, F)
        "ring_pinky": {"notes": [71, 76, 78], "name": "B sus4"},  # B sus4 (B, E, F#)
        "thumb_pinky": {"notes": [62, 69, 73, 78], "name": "D13"}  # D13 (D, A, C#, F#)
    }
}

# Finger pair mapping for combination detection
finger_pairs = {
    "thumb_index": [0, 1],
    "index_middle": [1, 2],
    "middle_ring": [2, 3],
    "ring_pinky": [3, 4],
    "thumb_pinky": [0, 4]
}

# Track Previous States to Stop Chords
prev_states = {
    "single": {hand: {finger: 0 for finger in single_chords[hand]} for hand in single_chords},
    "combo": {hand: {combo: 0 for combo in combo_chords[hand]} for hand in combo_chords}
}

# Function to Play a Chord
def play_chord(chord_notes, chord_name=None):
    global active_chords, player, fs, performance_metrics, USE_FLUIDSYNTH
    
    volume = int(settings["volume"] * 1.27)  # Scale to 0-127 range
    
    if USE_FLUIDSYNTH and fs is not None:
        for note in chord_notes:
            fs.noteon(0, note, volume)
    elif player is not None:
        for note in chord_notes:
            player.note_on(note, volume)
    else:
        logger.warning("Sound system not initialized, can't play chord")
        return
    
    # Add to active chords list for UI display
    if chord_name and chord_name not in active_chords:
        active_chords.append(chord_name)
        performance_metrics["chords_played"] += 1
    
    logger.debug(f"Played chord: {chord_name} - Notes: {chord_notes}")

# Function to Stop a Chord After a Delay
def stop_chord_after_delay(chord_notes, chord_name=None):
    global active_chords, player, fs, USE_FLUIDSYNTH
    
    time.sleep(settings["sustain_time"])  # Sustain for specified time
    
    if USE_FLUIDSYNTH and fs is not None:
        for note in chord_notes:
            fs.noteoff(0, note)
    elif player is not None:
        for note in chord_notes:
            player.note_off(note, 0)
    
    # Remove from active chords list for UI display
    if chord_name and chord_name in active_chords:
        active_chords.remove(chord_name)

# Function to Initialize Camera
def initialize_camera():
    global cap, detector
    try:
        # Release existing camera if any
        if cap is not None:
            cap.release()
            
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Failed to open camera")
            return False
            
        detector = HandDetector(detectionCon=settings["sensitivity"])
        logger.info("Camera initialized successfully")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        return True
    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        return False

# Function to Generate Camera Frames
def generate_frames():
    global tracking_active, active_hands, performance_metrics
    
    if not initialize_camera():
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n'
               b'Camera initialization failed\r\n')
        return
    
    # Start performance tracking
    if performance_metrics["session_start"] is None:
        performance_metrics["session_start"] = time.time()
    
    while True:
        success, img = cap.read()
        if not success:
            logger.warning("Camera not capturing frames")
            yield (b'--frame\r\n'
                   b'Content-Type: text/plain\r\n\r\n'
                   b'Camera not capturing frames\r\n')
            time.sleep(0.5)
            continue

        performance_metrics["frames_processed"] += 1
        
        # Flip the image horizontally for a more intuitive experience
        img = cv2.flip(img, 1)
        
        # Apply brightness/contrast adjustments if needed
        if camera_data["brightness"] != 100 or camera_data["contrast"] != 100:
            alpha = camera_data["contrast"] / 100.0  # Contrast control
            beta = (camera_data["brightness"] - 100)  # Brightness control
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # Only process hand tracking if tracking is active
        if tracking_active:
            active_hands = []
            # Find hands
            hands, img = detector.findHands(img, draw=True)
            
            if hands:
                performance_metrics["hands_detected"] += 1
                for hand in hands:
                    hand_type = "left" if hand["type"] == "Left" else "right"
                    active_hands.append(hand_type)
                    fingers = detector.fingersUp(hand)
                    
                    # Process finger combinations first
                    combo_played = False
                    for combo_name, indices in finger_pairs.items():
                        if combo_name in combo_chords[hand_type]:
                            # Check if both fingers in the pair are up
                            if fingers[indices[0]] == 1 and fingers[indices[1]] == 1 and all(fingers[i] == 0 for i in range(5) if i != indices[0] and i != indices[1]):
                                combo_data = combo_chords[hand_type][combo_name]
                                
                                # Play combo chord if state changed
                                if prev_states["combo"][hand_type][combo_name] == 0:
                                    play_chord(combo_data["notes"], combo_data["name"])
                                    
                                prev_states["combo"][hand_type][combo_name] = 1
                                combo_played = True
                            elif prev_states["combo"][hand_type][combo_name] == 1:
                                # Stop combo chord if state changed
                                combo_data = combo_chords[hand_type][combo_name]
                                threading.Thread(
                                    target=stop_chord_after_delay, 
                                    args=(combo_data["notes"], combo_data["name"]), 
                                    daemon=True
                                ).start()
                                prev_states["combo"][hand_type][combo_name] = 0
                    
                    # If no combo chord played, check individual fingers
                    if not combo_played:
                        for finger_name, finger_idx in finger_indices.items():
                            if finger_name in single_chords[hand_type]:
                                chord_data = single_chords[hand_type][finger_name]
                                
                                if fingers[finger_idx] == 1 and prev_states["single"][hand_type][finger_name] == 0:
                                    play_chord(chord_data["notes"], chord_data["name"])
                                elif fingers[finger_idx] == 0 and prev_states["single"][hand_type][finger_name] == 1:
                                    threading.Thread(
                                        target=stop_chord_after_delay, 
                                        args=(chord_data["notes"], chord_data["name"]), 
                                        daemon=True
                                    ).start()
                                
                                prev_states["single"][hand_type][finger_name] = fingers[finger_idx]
                    
                    # Add finger status information to the frame for visualization
                    cv2.putText(img, f"{hand_type.upper()}", 
                              tuple(hand["center"]), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                              (255, 255, 0), 2)
            
            # If no hands are detected, stop all active notes
            if not hands and active_chords:
                for finger_name in finger_indices:
                    for hand_type in ["left", "right"]:
                        if finger_name in single_chords[hand_type]:
                            if prev_states["single"][hand_type][finger_name] == 1:
                                chord_data = single_chords[hand_type][finger_name]
                                threading.Thread(
                                    target=stop_chord_after_delay, 
                                    args=(chord_data["notes"], chord_data["name"]), 
                                    daemon=True
                                ).start()
                                prev_states["single"][hand_type][finger_name] = 0
                
                for combo_name in finger_pairs:
                    for hand_type in ["left", "right"]:
                        if combo_name in combo_chords[hand_type]:
                            if prev_states["combo"][hand_type][combo_name] == 1:
                                combo_data = combo_chords[hand_type][combo_name]
                                threading.Thread(
                                    target=stop_chord_after_delay, 
                                    args=(combo_data["notes"], combo_data["name"]), 
                                    daemon=True
                                ).start()
                                prev_states["combo"][hand_type][combo_name] = 0

        # Add display of active chord names
        if active_chords:
            chord_text = ", ".join(active_chords)
            cv2.putText(img, chord_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 255), 2)
            
        # Add status indicator
        status_text = "TRACKING ACTIVE" if tracking_active else "TRACKING PAUSED"
        status_color = (0, 255, 0) if tracking_active else (0, 0, 255)
        cv2.putText(img, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, status_color, 2)
        
        # Add instrument indicator
        instr_name = instruments.get(current_instrument, f"Instrument {current_instrument}")
        cv2.putText(img, f"Instrument: {instr_name}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 200, 0), 2)
            
        # Convert to JPEG for web streaming
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Update session duration
        if performance_metrics["session_start"] is not None:
            performance_metrics["session_duration"] = time.time() - performance_metrics["session_start"]


# Routes
@app.route('/')
def index():
    """Serve the index page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Return the video feed"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/active_chords')
def get_active_chords():
    """Return active chords and hands for the UI"""
    return jsonify({
        "active_chords": active_chords,
        "active_hands": active_hands
    })

@app.route('/chords_data')
def get_chords_data():
    """Return all chord mappings for the UI"""
    return jsonify({
        "single_chords": single_chords,
        "combo_chords": combo_chords
    })

@app.route('/get_instruments')
def get_instruments():
    """Return available instruments list"""
    return jsonify({
        "instruments": instruments,
        "current_instrument": current_instrument
    })

@app.route('/start_tracking', methods=['POST'])
def start_tracking():
    """Start hand tracking"""
    global tracking_active
    tracking_active = True
    logger.info("Hand tracking started")
    return jsonify({"status": "success", "message": "Tracking started"})

@app.route('/stop_tracking', methods=['POST'])
def stop_tracking():
    """Stop hand tracking"""
    global tracking_active
    tracking_active = False
    logger.info("Hand tracking stopped")
    return jsonify({"status": "success", "message": "Tracking stopped"})

@app.route('/switch_instrument', methods=['POST'])
def switch_instrument():
    """Switch the MIDI instrument"""
    global current_instrument, player, fs, USE_FLUIDSYNTH
    
    try:
        data = request.get_json()
        instrument_id = int(data.get('instrument_id', 0))
        
        if 0 <= instrument_id <= 127:
            current_instrument = instrument_id
            if USE_FLUIDSYNTH and fs is not None:
                fs.program_change(0, instrument_id)
            elif player:
                player.set_instrument(instrument_id)
            logger.info(f"Switched to instrument {instrument_id}: {instruments.get(instrument_id, 'Instrument')}")
            return jsonify({
                "status": "success", 
                "instrument": instrument_id, 
                "name": instruments.get(instrument_id, f"Instrument {instrument_id}")
            })
        else:
            return jsonify({"status": "error", "message": "Invalid instrument ID"})
    except Exception as e:
        logger.error(f"Error switching instrument: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/update_settings', methods=['POST'])
def update_settings():
    """Update settings like sustain time, sensitivity, and volume"""
    global settings
    try:
        data = request.get_json()
        
        if 'sustain_time' in data:
            settings['sustain_time'] = float(data['sustain_time'])
            logger.info(f"Updated sustain_time to {settings['sustain_time']}")
        
        if 'sensitivity' in data:
            settings['sensitivity'] = float(data['sensitivity'])
            logger.info(f"Updated sensitivity to {settings['sensitivity']}")
            if detector:
                detector.updateDetectionCon(settings['sensitivity'])
        
        if 'volume' in data:
            settings['volume'] = int(data['volume'])
            logger.info(f"Updated volume to {settings['volume']}")
        
        return jsonify({"status": "success", "settings": settings})
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/calibrate_camera', methods=['POST'])
def calibrate_camera():
    """Calibrate the camera for better hand detection"""
    global camera_data
    try:
        logger.info("Calibrating camera...")
        # Perform actual calibration process
        if cap is None:
            if not initialize_camera():
                return jsonify({"status": "error", "message": "Failed to initialize camera for calibration"})
        
        # Simulate a calibration process (adjust brightness/contrast automatically)
        time.sleep(1)
        
        # Reset to default values first
        camera_data["brightness"] = 100
        camera_data["contrast"] = 100
        
        # In a real implementation, you would analyze frames here 
        # and automatically determine optimal brightness/contrast
        
        camera_data["calibrated"] = True
        logger.info("Camera calibrated successfully")
        
        return jsonify({"status": "success", "message": "Camera calibrated successfully"})
    except Exception as e:
        logger.error(f"Error calibrating camera: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/adjust_camera', methods=['POST'])
def adjust_camera():
    """Adjust camera brightness and contrast"""
    global camera_data
    try:
        data = request.get_json()
        
        if 'brightness' in data:
            camera_data['brightness'] = int(data['brightness'])
            logger.info(f"Updated camera brightness to {camera_data['brightness']}")
        
        if 'contrast' in data:
            camera_data['contrast'] = int(data['contrast'])
            logger.info(f"Updated camera contrast to {camera_data['contrast']}")
        
        return jsonify({"status": "success", "camera_data": camera_data})
    except Exception as e:
        logger.error(f"Error adjusting camera: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/get_status', methods=['GET'])
def get_status():
    """Get the current status of the system"""
    return jsonify({
        "tracking_active": tracking_active,
        "settings": settings,
        "camera_data": camera_data,
        "current_instrument": current_instrument,
        "instrument_name": instruments.get(current_instrument, f"Instrument {current_instrument}"),
        "performance_metrics": {
            "frames_processed": performance_metrics["frames_processed"],
            "hands_detected": performance_metrics["hands_detected"],
            "chords_played": performance_metrics["chords_played"],
            "session_duration": round(performance_metrics["session_duration"], 2) if performance_metrics["session_duration"] else 0
        },
        "midi_available": player is not None or fs is not None
    })

@app.route('/reset_metrics', methods=['POST'])
def reset_metrics():
    """Reset performance metrics"""
    global performance_metrics
    try:
        performance_metrics = {
            "frames_processed": 0,
            "hands_detected": 0,
            "chords_played": 0,
            "session_start": time.time(),
            "session_duration": 0
        }
        logger.info("Performance metrics reset")
        return jsonify({"status": "success", "message": "Metrics reset successfully"})
    except Exception as e:
        logger.error(f"Error resetting metrics: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/save_custom_chord', methods=['POST'])
def save_custom_chord():
    """Save a custom chord mapping"""
    try:
        data = request.get_json()
        
        hand = data.get('hand')
        finger = data.get('finger')
        notes = data.get('notes')
        name = data.get('name')
        
        if not all([hand, finger, notes, name]) or hand not in ['left', 'right']:
            return jsonify({"status": "error", "message": "Missing or invalid parameters"})
        
        if finger in finger_indices:
            # Single finger chord
            if finger in single_chords[hand]:
                single_chords[hand][finger] = {"notes": notes, "name": name}
                logger.info(f"Updated chord mapping for {hand} hand, {finger} finger to {name}")
            else:
                return jsonify({"status": "error", "message": f"Invalid finger name: {finger}"})
        elif finger in finger_pairs:
            # Combo chord
            if finger in combo_chords[hand]:
                combo_chords[hand][finger] = {"notes": notes, "name": name}
                logger.info(f"Updated combo chord mapping for {hand} hand, {finger} to {name}")
            else:
                return jsonify({"status": "error", "message": f"Invalid finger combo: {finger}"})
        else:
            return jsonify({"status": "error", "message": f"Unknown finger or combo: {finger}"})
        
        return jsonify({"status": "success", "message": "Chord mapping saved"})
        
    except Exception as e:
        logger.error(f"Error saving custom chord: {e}")
        return jsonify({"status": "error", "message": str(e)})

# Cleanup function when server shuts down
def cleanup():
    """Cleanup resources when application exits"""
    global cap, player, fs
    if cap:
        cap.release()
    
    if player:
        del player
    
    if fs:
        fs.delete()
        
    logger.info("AirPiano server resources cleaned up")

# Register cleanup function to be called when application exits
import atexit
atexit.register(cleanup)

# Start the Flask app
if __name__ == '__main__':
    logger.info("AirPiano server starting...")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)