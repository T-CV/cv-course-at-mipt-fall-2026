import cv2
import time
try:
    import dlib
except ImportError:
    raise RuntimeError("dlib не найден. Установить через pip install dlib")


detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Не удалось открыть вебкамеру")

print("Нажми 'q' для выхода")

prev_t = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 0)

    for r in faces:
        x1, y1, x2, y2 = r.left(), r.top(), r.right(), r.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    now = time.time()
    fps = 1.0 / max(now - prev_t, 1e-6)
    prev_t = now
    cv2.putText(frame, f"FPS: {fps:.1f}  Faces: {len(faces)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("HOG Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()