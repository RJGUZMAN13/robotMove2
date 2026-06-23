# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 17:16:26 2026

@author: rjguz
"""

import cv2
import mediapipe as mp
import snap7
import numpy as np
from snap7.util import set_bool

# =====================================================
# CONFIGURACION PLC
# =====================================================

PLC_IP = "192.168.11.1"
RACK = 0
SLOT = 1
DB_NUMBER = 1

# =====================================================
# COLORES
# =====================================================

BG = (25, 25, 25)
PANEL = (35, 35, 35)

WHITE = (255, 255, 255)
GREEN = (0, 220, 0)
RED = (0, 0, 220)
BLUE = (220, 120, 0)
YELLOW = (0, 255, 255)
GRAY = (90, 90, 90)

# =====================================================
# PLC
# =====================================================

client = snap7.client.Client()
plc_connected = False


try:
    print(f"Intentando conectar a {PLC_IP}...")
    
    client.connect(PLC_IP, RACK, SLOT)

    print("Conectado:", client.get_connected())

    plc_connected = client.get_connected()

except Exception as e:

    print("ERROR DE CONEXION PLC")
    print(type(e))
    print(e)

    plc_connected = False

# =====================================================
# MEDIAPIPE
# =====================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =====================================================
# FUNCION DEDOS
# =====================================================

def fingers_up(hand):

    tips = [4, 8, 12, 16, 20]

    fingers = []

    thumb = 1 if hand.landmark[4].x < hand.landmark[3].x else 0
    fingers.append(thumb)

    for i in range(1, 5):

        if hand.landmark[tips[i]].y < hand.landmark[tips[i]-2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

# =====================================================
# CAMARA
# =====================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("No se pudo abrir la camara")

cv2.namedWindow(
    "Industrial Hand Robot Control",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Industrial Hand Robot Control",
    1600,
    900
)

cv2.namedWindow(
    "RJG Vision Tracking",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "RJG Vision Tracking",
    900,
    900
)

# =====================================================
# LOOP
# =====================================================

while True:

    ret, frame = cap.read()
    

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    
    skeleton_view = np.zeros(
    (h, w, 3),
    dtype=np.uint8
    )
    
    # Fondo profesional
    
    cv2.rectangle(
        skeleton_view,
        (0, 0),
        (w, h),
        (15, 15, 15),
        -1
    )
    
    # Grid tecnológica
    
    for x in range(0, w, 50):
        cv2.line(
            skeleton_view,
            (x, 0),
            (x, h),
            (25, 25, 25),
            1
        )
    
    for y in range(0, h, 50):
        cv2.line(
            skeleton_view,
            (0, y),
            (w, y),
            (25, 25, 25),
            1
        )

    dashboard_width = 420

    frame = cv2.copyMakeBorder(
        frame,
        0,
        0,
        0,
        dashboard_width,
        cv2.BORDER_CONSTANT,
        value=PANEL
    )

    camera_width = w

    center_x = camera_width // 2

    rgb = cv2.cvtColor(
        frame[:, :camera_width],
        cv2.COLOR_BGR2RGB
    )

    result = hands.process(rgb)

    gripper = False

    left = False
    center = False
    right = False

    tracking_status = "NO HAND"

    robot_position = "CENTER"

    hand_x = center_x
    offset = 0

    # =================================================
    # DETECCION
    # =================================================

    if result.multi_hand_landmarks:

        tracking_status = "ACTIVE"

        hand = result.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame[:, :camera_width],
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        mp_draw.draw_landmarks(
            skeleton_view,
            hand,
            mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(
                color=(0,255,255),
                thickness=3,
                circle_radius=4
            ),
            mp_draw.DrawingSpec(
                color=(255,255,255),
                thickness=2
            )
        )
        

        wrist = hand.landmark[0]

        hand_x = int(wrist.x * camera_width)

        offset = hand_x - center_x

        fingers = fingers_up(hand)

        total_fingers = sum(fingers)

        if total_fingers >= 4:
            gripper = False
            gripper_text = "OPEN"

        elif total_fingers <= 1:
            gripper = True
            gripper_text = "CLOSE"

        else:
            gripper_text = "PARTIAL"

        dead_zone = 80

        if hand_x < center_x - dead_zone:

            left = True
            robot_position = "LEFT"

        elif hand_x > center_x + dead_zone:

            right = True
            robot_position = "RIGHT"

        else:

            center = True
            robot_position = "CENTER"

        cv2.circle(
            frame,
            (hand_x, h // 2),
            12,
            YELLOW,
            -1
        )

    else:

        gripper_text = "UNKNOWN"

    # =================================================
    # ZONAS
    # =================================================

    dead_zone = 80

    cv2.rectangle(
        frame,
        (0, 0),
        (center_x - dead_zone, h),
        (20, 20, 70),
        2
    )

    cv2.rectangle(
        frame,
        (center_x - dead_zone, 0),
        (center_x + dead_zone, h),
        (20, 70, 20),
        2
    )

    cv2.rectangle(
        frame,
        (center_x + dead_zone, 0),
        (camera_width, h),
        (70, 20, 20),
        2
    )

    cv2.putText(
        frame,
        "LEFT",
        (40, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        RED,
        2
    )

    cv2.putText(
        frame,
        "CENTER",
        (center_x - 60, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        GREEN,
        2
    )

    cv2.putText(
        frame,
        "RIGHT",
        (camera_width - 120, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        BLUE,
        2
    )

    # =================================================
    # PLC
    # =================================================

    try:
    
        data = bytearray(1)
    
        set_bool(data, 0, 0, right)
        set_bool(data, 0, 1, center)
        set_bool(data, 0, 2, left)
        set_bool(data, 0, 3, gripper)
    
        print(
            f"DERECHA={right} | "
            f"CENTRO={center} | "
            f"IZQUIERDA={left} | "
            f"GRIPPER={gripper}"
        )
    
        print("BYTE:", list(data))
    
        client.db_write(
            DB_NUMBER,
            26,
            data
        )
    
        plc_connected = client.get_connected()
    
    except Exception as e:
    
        print("ERROR PLC:")
        print(e)
    
        plc_connected = False

    # =================================================
    # DASHBOARD
    # =================================================

    px = camera_width + 20

    cv2.putText(
        frame,
        "RJG HAND CONTROL",
        (px, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        WHITE,
        2
    )

    cv2.line(
        frame,
        (camera_width, 60),
        (camera_width + dashboard_width, 60),
        GRAY,
        2
    )

    cv2.putText(
        frame,
        f"PLC: {'CONNECTED' if plc_connected else 'OFFLINE'}",
        (px, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        GREEN if plc_connected else RED,
        2
    )

    cv2.putText(
        frame,
        f"TRACKING: {tracking_status}",
        (px, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        WHITE,
        2
    )

    cv2.putText(
        frame,
        "GRIPPER",
        (px, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        WHITE,
        2
    )

    cv2.putText(
        frame,
        gripper_text,
        (px, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        GREEN if gripper_text == "OPEN" else RED,
        3
    )

    cv2.putText(
        frame,
        "COMMAND",
        (px, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        WHITE,
        2
    )

    color_cmd = GREEN

    if robot_position == "LEFT":
        color_cmd = RED

    elif robot_position == "RIGHT":
        color_cmd = BLUE

    cv2.putText(
        frame,
        robot_position,
        (px, 390),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        color_cmd,
        4
    )

    cv2.putText(
        frame,
        f"HAND X : {hand_x}",
        (px, 470),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        WHITE,
        2
    )

    cv2.putText(
        frame,
        f"CENTER : {center_x}",
        (px, 510),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        WHITE,
        2
    )

    cv2.putText(
        frame,
        f"OFFSET : {offset}",
        (px, 550),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        WHITE,
        2
    )

    cv2.putText(
        frame,
        f"LEFT   : {left}",
        (px, 620),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        WHITE,
        2
    )

    cv2.putText(
        frame,
        f"CENTER : {center}",
        (px, 660),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        WHITE,
        2
    )

    cv2.putText(
        frame,
        f"RIGHT  : {right}",
        (px, 700),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        WHITE,
        2
    )

    # =================================================
    # BARRA POSICION
    # =================================================

    bar_y = h - 50

    cv2.line(
        frame,
        (50, bar_y),
        (camera_width - 50, bar_y),
        WHITE,
        4
    )

    marker_x = max(
        50,
        min(hand_x, camera_width - 50)
    )

    cv2.circle(
        frame,
        (marker_x, bar_y),
        15,
        YELLOW,
        -1
    )
    
    
    
    # =================================================
    # VENTANA TRACKING PROFESIONAL
    # =================================================
    
    cv2.putText(
        skeleton_view,
        "RJG AI ROBOT CONTROL",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )
    
    cv2.line(
        skeleton_view,
        (20,55),
        (420,55),
        (0,255,255),
        2
    )
    
    cv2.putText(
        skeleton_view,
        f"TRACKING : {tracking_status}",
        (20,100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )
    
    # COLOR SEGUN POSICION
    
    position_color = YELLOW
    
    if robot_position == "LEFT":
        position_color = RED
    
    elif robot_position == "RIGHT":
        position_color = BLUE
    
    elif robot_position == "CENTER":
        position_color = YELLOW
    
    cv2.putText(
        skeleton_view,
        f"POSITION : {robot_position}",
        (20,140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        position_color,
        2
    )
    
    # COLOR SEGUN ESTADO DEL GRIPPER
    
    gripper_color = WHITE
    
    if gripper_text == "OPEN":
        gripper_color = GREEN
    
    elif gripper_text == "CLOSE":
        gripper_color = RED
    
    elif gripper_text == "PARTIAL":
        gripper_color = YELLOW
    
    cv2.putText(
        skeleton_view,
        f"GRIPPER : {gripper_text}",
        (20,180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        gripper_color,
        2
    )
    
    cv2.putText(
        skeleton_view,
        f"FINGERS : {total_fingers if tracking_status == 'ACTIVE' else 0}",
        (20,220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    # =================================================
    # MOSTRAR
    # =================================================

    cv2.imshow(
        "Industrial Hand Robot Control",
        frame
    )
    
    cv2.imshow(
        "RJG Vision Tracking",
        skeleton_view
    )
    

    if cv2.getWindowProperty(
        "Industrial Hand Robot Control",
        cv2.WND_PROP_VISIBLE
    ) < 1:
        break
    
    
    if cv2.getWindowProperty(
        "RJG Vision Tracking",
        cv2.WND_PROP_VISIBLE
    ) < 1:
        break

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

# =====================================================
# CIERRE
# =====================================================

cap.release()
cv2.destroyAllWindows()

try:
    if client.get_connected():
        client.disconnect()
except:
    pass