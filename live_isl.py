"""
Live ISL Sign-to-Speech — Standalone
Uses OpenCV, MediaPipe Tasks API, macOS TTS.
"""

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import subprocess
import time
import threading
import os
import collections

# ── MediaPipe Tasks Setup ──
try:
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    
    # Setup Face Landmarker
    face_base = python.BaseOptions(model_asset_path='face_landmarker.task')
    face_options = vision.FaceLandmarkerOptions(
        base_options=face_base, 
        output_face_blendshapes=True,
        num_faces=1
    )
    face_detector = vision.FaceLandmarker.create_from_options(face_options)
except Exception as e:
    print(f"Could not load Landmark models. Ensure .task files exist. Error: {e}")
    exit(1)

# ── State ──
sentence = []
last_sign = ""
last_sign_time = 0
hold_start = 0
HOLD_MS = 600
COOLDOWN_MS = 1500

# Tracking properties
wrist_x_history = collections.deque(maxlen=15)
wrist_history_h1 = collections.deque(maxlen=15)  # hand 1 wrist x for ISL deaf clap
wrist_history_h2 = collections.deque(maxlen=15)  # hand 2 wrist x for ISL deaf clap
last_clap_time = 0
last_expression = "Neutral"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

# ── TTS (macOS `say`) ──
def speak(text):
    def _say():
        subprocess.run(["say", "-v", "Samantha", text], check=False)
    threading.Thread(target=_say, daemon=True).start()



def distance2d(p1, p2, w, h):
    return ((p1.x * w - p2.x * w)**2 + (p1.y * h - p2.y * h)**2) ** 0.5

def get_finger_states(lm, w, h):
    """Finger extension using PIP joints (mid-finger) — much more reliable than MCP."""
    wrist = lm[0]

    # Thumb: extended if tip (4) is far from index base (5)
    # When thumb sticks out, tip moves away from index finger base
    # When tucked (fist), tip stays near index base
    thumb = distance2d(lm[4], lm[5], w, h) > distance2d(lm[3], lm[5], w, h)

    # Fingers: tip further from wrist than PIP joint = extended
    # PIP joints (6,10,14,18) are mid-finger — strong signal
    # When extended: tip >> PIP distance from wrist
    # When curled: tip ≈ or < PIP distance from wrist
    index  = distance2d(lm[8],  wrist, w, h) > distance2d(lm[6],  wrist, w, h)
    middle = distance2d(lm[12], wrist, w, h) > distance2d(lm[10], wrist, w, h)
    ring   = distance2d(lm[16], wrist, w, h) > distance2d(lm[14], wrist, w, h)
    pinky  = distance2d(lm[20], wrist, w, h) > distance2d(lm[18], wrist, w, h)

    return [thumb, index, middle, ring, pinky]

def classify_gesture(results, w, h):
    # ── ISL DEAF CLAP: Both hands raised, palms forward, shaking ──
    global last_clap_time, wrist_history_h1, wrist_history_h2
    if len(results.hand_landmarks) == 2:
        lm1 = results.hand_landmarks[0]
        lm2 = results.hand_landmarks[1]
        f1 = get_finger_states(lm1, w, h)
        f2 = get_finger_states(lm2, w, h)
        both_open = sum(f1[1:]) >= 3 and sum(f2[1:]) >= 3

        # Both hands raised (wrists in upper portion of frame)
        both_raised = lm1[0].y < 0.6 and lm2[0].y < 0.6

        # Track wrist x positions for shake detection
        wrist_history_h1.append(lm1[0].x * w)
        wrist_history_h2.append(lm2[0].x * w)

        if both_open and both_raised:
            now = time.time() * 1000
            # Detect shaking: lateral oscillation in at least one hand
            if len(wrist_history_h1) >= 6 and len(wrist_history_h2) >= 6:
                range1 = max(wrist_history_h1) - min(wrist_history_h1)
                range2 = max(wrist_history_h2) - min(wrist_history_h2)
                # Either hand shaking is enough (lowered threshold)
                if (range1 > (0.025 * w) or range2 > (0.025 * w)) and (now - last_clap_time > 1200):
                    last_clap_time = now
                    wrist_history_h1.clear()
                    wrist_history_h2.clear()
                    return "CLAP", 1.0
            # Two open raised hands but no shake yet — don't fall through to single-hand
            return "CLAP_PENDING", 0.0
        else:
            wrist_history_h1.clear()
            wrist_history_h2.clear()

    if not results.hand_landmarks:
        return None, 0

    lm = results.hand_landmarks[0]
    thumb, index, middle, ring, pinky = get_finger_states(lm, w, h)
    non_thumb_count = sum([index, middle, ring, pinky])

    # ── OK: thumb tip touches index tip, making O shape. Other 3 fingers up ──
    thumb_index_dist = distance2d(lm[4], lm[8], w, h)
    if thumb_index_dist < (0.08 * w) and middle and ring and pinky:
        return "OK", 0.85

    # ── LOVE (ILY): thumb + index + pinky extended, middle & ring curled ──
    if thumb and index and not middle and not ring and pinky:
        return "LOVE", 0.88

    # ── CALL: thumb + pinky only ──
    if thumb and not index and not middle and not ring and pinky:
        return "CALL", 0.85

    # ── Thumb-to-palm distance (reused below) ──
    thumb_to_palm = distance2d(lm[4], lm[9], w, h)  # thumb tip to middle MCP

    # ── Numbers: Do NOT check thumb (unreliable when tucked) ──
    if non_thumb_count == 1 and index:
        return "ONE", 0.85
    if non_thumb_count == 2 and index and middle:
        return "TWO", 0.85
    if non_thumb_count == 3 and index and middle and ring:
        return "THREE", 0.8

    # ── FIVE / HELLO: all 4 fingers up + thumb spread — CHECK BEFORE FOUR ──
    # Thumb clearly away from palm = open hand, not tucked
    if non_thumb_count == 4 and thumb_to_palm >= (0.09 * w):
        return "FIVE_OR_HELLO", 0.9

    # ── FOUR: all 4 fingers up, thumb clearly tucked against palm ──
    if non_thumb_count == 4 and thumb_to_palm < (0.09 * w):
        return "FOUR", 0.8

    # ── GOOD vs BAD vs YES: All 4 non-thumb fingers curled ──
    if non_thumb_count == 0:
        if thumb_to_palm > (0.08 * w):
            if lm[4].y < lm[0].y:
                return "GOOD", 0.88
            else:
                return "BAD", 0.85
        else:
            return "YES", 0.8

    return None, 0

def apply_hud(frame, sign_name, expression):
    h, w, c = frame.shape

    # Top bar with sign detection and face expression
    cv2.rectangle(frame, (0, 0), (w, 50), (15, 15, 30), -1)

    det_text = sign_name if sign_name else "No sign"
    cv2.putText(frame, f"Sign: {det_text}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 144, 30), 2)

    cv2.putText(frame, f"Sent: {' '.join(sentence[-5:])}", (230, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Facial Expression — top right
    cv2.putText(frame, f"Face: {expression}", (w - 180, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (144, 255, 30), 2)

    return frame

def get_face_expression(face_results):
    if not face_results.face_blendshapes: return "Neutral"
    shapes = face_results.face_blendshapes[0]
    
    cat_map = {c.category_name: c.score for c in shapes}
    
    smile = (cat_map.get('mouthSmileLeft', 0) + cat_map.get('mouthSmileRight', 0)) / 2
    frown = (cat_map.get('mouthFrownLeft', 0) + cat_map.get('mouthFrownRight', 0)) / 2
    jaw_open = cat_map.get('jawOpen', 0)
    brow_inner_up = cat_map.get('browInnerUp', 0)
    brow_down = (cat_map.get('browDownLeft', 0) + cat_map.get('browDownRight', 0)) / 2
    squint = (cat_map.get('eyeSquintLeft', 0) + cat_map.get('eyeSquintRight', 0)) / 2
    brow_asym = abs(cat_map.get('browOuterUpLeft', 0) - cat_map.get('browOuterUpRight', 0))
    
    # Excited: Wide smile with opened jaw
    if smile > 0.35 and jaw_open > 0.15: return "Excited"
    
    # Happy: Regular smile
    if smile > 0.3: return "Happy"
    
    # Sad: Frowning sometimes coupled with inner brows raised
    if frown > 0.15 and brow_inner_up > 0.15: return "Sad"
    if frown > 0.35: return "Sad"
    
    # Confused: Squinting + brow down (intense concentration) OR asymmetrical eyebrow (raised one side)
    if squint > 0.2 and brow_down > 0.2: return "Confused"
    if brow_asym > 0.2: return "Confused"
    
    return "Neutral"

def main():
    global last_sign, last_sign_time, hold_start, sentence, wrist_x_history, last_expression
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    
    print("\n╔══════════════════════════════════════╗")
    print("║   ISL Sign-to-Speech (Standalone)    ║")
    print("╠══════════════════════════════════════╣")
    print("║  NOTE: Click the VIDEO WINDOW to use ║")
    print("║  keys (Q/S/C), not the terminal!     ║")
    print("║  Q = Quit  | S = Speak  | C = Clear  ║")
    print("╚══════════════════════════════════════╝\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # ── Mediapipe Inference ──
        hand_results = detector.detect(mp_image)
        face_results = face_detector.detect(mp_image)
        
        h, w, _ = frame.shape
        
        # Face evaluation
        expression = get_face_expression(face_results)
        if expression != last_expression and expression != "Neutral":
            log_line = f"{time.strftime('%H:%M:%S')} - {expression}\n"
            try:
                with open("facial_expressions.log", "a") as f:
                    f.write(log_line)
            except: pass
            last_expression = expression
            print(f"  🧑 Face: {expression}")

        sign_name = None

        if hand_results.hand_landmarks:
            lm = hand_results.hand_landmarks[0]
            wrist_x_history.append(lm[0].x * w)
            
            sign_name, confidence = classify_gesture(hand_results, w, h)
            
            # Motion check for HELLO vs FIVE
            if sign_name == "FIVE_OR_HELLO":
                if len(wrist_x_history) >= 8 and (max(wrist_x_history) - min(wrist_x_history)) > (0.07 * w):
                    sign_name = "HELLO"
                else:
                    sign_name = "FIVE"

            for lmarks in hand_results.hand_landmarks:
                for connection in HAND_CONNECTIONS:
                    p1, p2 = lmarks[connection[0]], lmarks[connection[1]]
                    cv2.line(frame, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (255, 144, 30), 2) 
                for point in lmarks:
                    cv2.circle(frame, (int(point.x * w), int(point.y * h)), 5, (255, 144, 30), -1)

        now = time.time() * 1000
        # CLAP_PENDING = two hands in position, waiting for shake. Don't register.
        if sign_name == "CLAP_PENDING":
            sign_name = None
        if sign_name:
            if sign_name == last_sign:
                if (now - hold_start > HOLD_MS) and (now - last_sign_time > COOLDOWN_MS):
                    sentence.append(sign_name)
                    last_sign_time = now
                    speak(sign_name)
                    print(f"  ✅ {sign_name}  |  Sent: {' '.join(sentence)}")
            else:
                last_sign = sign_name
                hold_start = now
        else:
            last_sign = ""

        try:
            frame = apply_hud(frame, sign_name, expression)
        except Exception as e:
            pass

        cv2.imshow("ISL Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'): break
        elif (key == ord('s') or key == ord('S')) and sentence:
            full = " ".join(sentence)
            print(f"  🔊 {full}")
            speak(full)
        elif key == ord('c') or key == ord('C'):
            sentence = []
            print("  🗑  Cleared")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
