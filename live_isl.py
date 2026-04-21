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
    wrist = lm[0]
    thumb = distance2d(lm[4], lm[17], w, h) > distance2d(lm[3], lm[17], w, h)
    index  = distance2d(lm[8], wrist, w, h) > distance2d(lm[5], wrist, w, h)
    middle = distance2d(lm[12], wrist, w, h) > distance2d(lm[9], wrist, w, h)
    ring   = distance2d(lm[16], wrist, w, h) > distance2d(lm[13], wrist, w, h)
    pinky  = distance2d(lm[20], wrist, w, h) > distance2d(lm[17], wrist, w, h)
    return [thumb, index, middle, ring, pinky]

def classify_gesture(results, w, h):
    # Two hands logic for Namaste
    if len(results.hand_landmarks) == 2:
        lm1 = results.hand_landmarks[0]
        lm2 = results.hand_landmarks[1]
        dist_wrists = distance2d(lm1[0], lm2[0], w, h)
        dist_index = distance2d(lm1[8], lm2[8], w, h)
        if dist_wrists < (0.2 * w) and dist_index < (0.2 * w):
            return "NAMASTE", 1.0

    if not results.hand_landmarks:
        return None, 0

    # Fallback to Single Hand
    lm = results.hand_landmarks[0]
    thumb, index, middle, ring, pinky = get_finger_states(lm, w, h)
    all_ext = all([thumb, index, middle, ring, pinky])
    all_curl = not any([thumb, index, middle, ring, pinky])

    if distance2d(lm[4], lm[8], w, h) < (0.06 * w) and middle and ring and pinky: return "OK", 0.82
    if thumb and index and not middle and not ring and pinky: return "LOVE", 0.88
    if thumb and not index and not middle and not ring and pinky: return "CALL", 0.85

    if not thumb and index and not middle and not ring and not pinky: return "ONE", 0.85
    if not thumb and index and middle and not ring and not pinky: return "TWO", 0.85
    if not thumb and index and middle and ring and not pinky: return "THREE", 0.8
    if not thumb and index and middle and ring and pinky: return "FOUR", 0.8
    
    if thumb and not index and not middle and not ring and not pinky:
        if lm[4].y < lm[0].y: return "GOOD", 0.88
        else: return "BAD", 0.85

    # If all fingers are extended, it's either HELLO (moving) or FIVE (stable).
    # We resolve this in the main loop using tracking.
    if all_ext: return "FIVE_OR_HELLO", 0.9
    
    if all_curl: return "YES", 0.8

    return None, 0

# Emoji Renderer for OpenCV
legend_banner = None

def overlay_transparent(background, overlay, x, y):
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width: w = background_width - x
    if y + h > background_height: h = background_height - y

    overlay = overlay[0:h, 0:w]
    if overlay.shape[2] == 4:
        alpha = overlay[..., 3] / 255.0
        for c in range(0, 3):
            background[y:y+h, x:x+w, c] = (alpha * overlay[..., c] + (1 - alpha) * background[y:y+h, x:x+w, c])
    else:
        background[y:y+h, x:x+w] = overlay
    return background

def apply_hud(frame, sign_name, expression):
    global legend_banner
    h, w, c = frame.shape
    banner_height = 100
    banner_y = h - banner_height

    if legend_banner is None or legend_banner.shape[1] != w:
        legend_banner = np.full((banner_height, w, 3), (30, 20, 20), dtype=np.uint8)
        
        legend_items = [
            ("ONE", "ONE"), ("TWO", "TWO"), ("THREE", "THREE"), ("FOUR", "FOUR"), ("FIVE", "FIVE"), ("HELLO", "HELLO"), ("NAMASTE", "NAMASTE"),
            ("GOOD", "GOOD"), ("BAD", "BAD"), ("YES", "YES"), ("LOVE", "LOVE"), ("CALL", "CALL"), ("OK", "OK")
        ]
        
        column_width = max(110, w // 7)
        x_offset = 10
        y_offset = 5
        
        for i, (name, text) in enumerate(legend_items):
            if i == 7:
                x_offset = 10
                y_offset += 45
            
            emoji_path = f"assets/emojis/{name}.png"
            if os.path.exists(emoji_path):
                emoji_img = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
                if emoji_img is not None:
                    emoji_img = cv2.resize(emoji_img, (32, 32))
                    legend_banner = overlay_transparent(legend_banner, emoji_img, x_offset, y_offset)
            
            cv2.putText(legend_banner, text, (x_offset + 40, y_offset + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            x_offset += column_width

    cv2.rectangle(frame, (0, 0), (w, 50), (15, 15, 30), -1)
    
    det_text = sign_name if sign_name else "No sign"
    cv2.putText(frame, f"Sign: {det_text}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 144, 30), 2)
    
    cv2.putText(frame, f"Sent: {' '.join(sentence[-5:])}", (230, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show Facial Expression top right
    cv2.putText(frame, f"Face: {expression}", (w - 180, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (144, 255, 30), 2)

    frame[banner_y:h, 0:w] = legend_banner
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
                if len(wrist_x_history) == 15 and (max(wrist_x_history) - min(wrist_x_history)) > (0.05 * w):
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
