import cv2
import mediapipe as mp
import numpy as np
import os

# ===== CONFIG =====
DATA_PATH = "dataset"
SAMPLES_PER_CLASS = 400

labels = [
"HELLO","GOODBYE","THANK_YOU","GOOD","BAD","LOVE","WAIT","SORRY",
"PLEASE","HELP","WATER","FOOD","SLEEP","BATHROOM","PAIN","I","NEED",
"NO_HAND"   # 👈 جديد
]

# ===== INIT =====
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# create folders
for label in labels:
    os.makedirs(os.path.join(DATA_PATH, label), exist_ok=True)

cap = cv2.VideoCapture(0)

print("\n📌 Available Gestures:")
for i, l in enumerate(labels):
    print(f"{i} -> {l}")

while True:
    choice = input(f"\n👉 Enter gesture number (0-{len(labels)-1}) or (q) to quit : ")

    if choice.lower() == 'q':
        break

    if not choice.isdigit() or int(choice) >= len(labels):
        print("❌ Invalid choice")
        continue

    label = labels[int(choice)]
    print(f"\n🎯 Selected: {label}")
    print("👉 Press 'S' to start collecting...")

    count = 0
    paused = False

    # wait for S
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f"{label} | Press S", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Collecting", frame)

        key = cv2.waitKey(1)
        if key == ord('s'):
            print("🚀 Collecting started...")
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # ===== COLLECT LOOP =====
    while count < SAMPLES_PER_CLASS:
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        data = []

        # ===== لو فيه إيد =====
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # رسم النقاط
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y, lm.z])

        # ===== لو مفيش إيد =====
        else:
            # 👇 ده أهم سطر لـ NO_HAND
            data = [0.0] * 63

        # ===== حفظ الداتا =====
        if not paused:
            np.save(f"{DATA_PATH}/{label}/{count}.npy", data)
            count += 1

        # ===== UI =====
        status = "PAUSED" if paused else "COLLECTING"

        cv2.putText(frame, f"{label}: {count}/{SAMPLES_PER_CLASS}", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.putText(frame, f"Status: {status}", (20,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.putText(frame, "P=Pause | C=Continue | Q=Exit", (20,150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Collecting", frame)

        key = cv2.waitKey(1)

        if key == ord('p'):
            paused = True
            print("⏸️ Paused...")

        elif key == ord('c'):
            paused = False
            print("▶️ Continue...")

        elif key == ord('q'):
            break

    print(f"✅ Done collecting {label}")

cap.release()
cv2.destroyAllWindows()