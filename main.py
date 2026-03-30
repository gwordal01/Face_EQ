import cv2
import mediapipe as mp
import pygame
import numpy as np
import math
import sys
import os

# ============================================================
# SETUP
# ============================================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: No webcam found.")
    sys.exit(1)

ret, test_frame = cap.read()
FRAME_H, FRAME_W = test_frame.shape[:2]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize audio
pygame.mixer.init(frequency=44100)

TRACK_FILE = "track.mp3"
if os.path.exists(TRACK_FILE):
    pygame.mixer.music.load(TRACK_FILE)
    # Get track length accurately using Sound object for limit handling
    sound = pygame.mixer.Sound(TRACK_FILE)
    TRACK_LENGTH = sound.get_length()
    print(f"Loaded: {TRACK_FILE} ({TRACK_LENGTH:.1f}s)")
else:
    print(f"ERROR: '{TRACK_FILE}' not found!")
    sys.exit(1)

# Landmarks
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
FOREHEAD = 10

def estimate_head_pose(landmarks, w, h):
    nose = landmarks[NOSE_TIP]
    left_eye = landmarks[LEFT_EYE_OUTER]
    right_eye = landmarks[RIGHT_EYE_OUTER]
    forehead = landmarks[FOREHEAD]
    chin = landmarks[CHIN]

    # YAW
    eye_mid_x = (left_eye.x + right_eye.x) / 2
    eye_distance = abs(right_eye.x - left_eye.x)
    yaw = ((nose.x - eye_mid_x) / eye_distance) * 60 if eye_distance > 0 else 0

    # PITCH
    nose_to_chin = chin.y - nose.y
    forehead_to_nose = nose.y - forehead.y
    pitch = ((nose_to_chin / forehead_to_nose) - 1.0) * 40 if forehead_to_nose > 0 else 0

    return yaw, pitch

# ============================================================
# TUNING (TODO #1 & #2)
# ============================================================
YAW_DEAD_ZONE = 6.0      # Degrees: wider zone = more stability
SCRUB_SENSITIVITY = 0.4  # Multiplier for how "heavy" the wheel feels

PITCH_DEAD_ZONE = 4.0    # Degrees
MIN_SPEED = 0.5
MAX_SPEED = 2.5          # Pushing it a bit for "chipmunk" effects

# ============================================================
# STATE
# ============================================================
is_playing = False
track_position = 0.0
playback_speed = 1.0
smooth_yaw, smooth_pitch = 0.0, 0.0
SMOOTHING = 0.25 # Lower is smoother, higher is more responsive

# ============================================================
# MAIN LOOP
# ============================================================
pygame.mixer.music.play()
is_playing = True

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        raw_yaw, raw_pitch = estimate_head_pose(landmarks, FRAME_W, FRAME_H)

        # Smooth Pose
        smooth_yaw += SMOOTHING * (raw_yaw - smooth_yaw)
        smooth_pitch += SMOOTHING * (raw_pitch - smooth_pitch)

        # ---- SCRUBBING (TODO #1) ----
        if abs(smooth_yaw) > YAW_DEAD_ZONE:
            # Calculate deflection past deadzone
            direction = 1 if smooth_yaw > 0 else -1
            deflection = abs(smooth_yaw) - YAW_DEAD_ZONE
            # Quadratic scaling: the further you turn, the faster it scrubs
            scrub_delta = (deflection ** 1.5) * 0.01 * SCRUB_SENSITIVITY * direction
            
            track_position += scrub_delta
            track_position = max(0, min(TRACK_LENGTH, track_position))
            pygame.mixer.music.set_pos(track_position)

        # ---- SPEED (TODO #2) ----
        if abs(smooth_pitch) > PITCH_DEAD_ZONE:
            # Map -20/+20 degrees to speed range
            pitch_norm = np.clip(smooth_pitch / 20.0, -1, 1)
            if pitch_norm > 0:
                # Speeding up
                playback_speed = 1.0 + (pitch_norm * (MAX_SPEED - 1.0))
            else:
                # Slowing down
                playback_speed = 1.0 + (pitch_norm * (1.0 - MIN_SPEED))
        else:
            playback_speed = 1.0

        # ---- VISUAL INDICATOR (TODO #3) ----
        g_center = (FRAME_W - 100, 100)
        g_radius = 60
        # Draw Gauge Background
        cv2.circle(frame, g_center, g_radius, (30, 30, 30), -1)
        cv2.circle(frame, g_center, g_radius, (0, 255, 255), 2)
        # Draw Crosshair
        cv2.line(frame, (g_center[0]-g_radius, g_center[1]), (g_center[0]+g_radius, g_center[1]), (60, 60, 60), 1)
        cv2.line(frame, (g_center[0], g_center[1]-g_radius), (g_center[0], g_center[1]+g_radius), (60, 60, 60), 1)
        
        # Position Dot
        dot_x = int(g_center[0] + (smooth_yaw / 30.0) * g_radius)
        dot_y = int(g_center[1] - (smooth_pitch / 20.0) * g_radius)
        
        # Color dot: Yellow if active, Grey if in deadzone
        active = abs(smooth_yaw) > YAW_DEAD_ZONE or abs(smooth_pitch) > PITCH_DEAD_ZONE
        dot_col = (0, 255, 255) if active else (100, 100, 100)
        cv2.circle(frame, (dot_x, dot_y), 8, dot_col, -1)

    # UI Overlay
    status_str = f"MODE: {'PLAY' if is_playing else 'PAUSE'}"
    cv2.putText(frame, status_str, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Speed: {playback_speed:.2f}x", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Pos: {track_position:.1f}s / {TRACK_LENGTH:.1f}s", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if abs(smooth_yaw) > YAW_DEAD_ZONE:
        col = (0, 165, 255) if smooth_yaw > 0 else (255, 0, 0)
        txt = ">> SCRUBBING FWD" if smooth_yaw > 0 else "<< SCRUBBING REW"
        cv2.putText(frame, txt, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

    cv2.imshow("BuildCored FaceEQ", frame)

    # Key Handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord(' '):
        is_playing = not is_playing
        pygame.mixer.music.pause() if not is_playing else pygame.mixer.music.unpause()
    elif key == ord('r'):
        track_position = 0.0
        pygame.mixer.music.play()

    if is_playing:
        # Note: Pygame mixer doesn't actually change the playback pitch/speed 
        # of a music stream in real-time. To simulate the "concept," we track 
        # position logic, but for true pitch shifting, one would use 
        # Sound.play() with frequency mods or a more robust library like librosa.
        track_position += (1/30.0) * playback_speed
        track_position = min(track_position, TRACK_LENGTH)

cap.release()
cv2.destroyAllWindows()