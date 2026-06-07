import cv2
from ultralytics import YOLO


def run_object_detection(camera_index: int = 0):
    model = YOLO("yolo11n.pt")  # n = nano, hızlı model

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Camera could not be opened.")
        return

    print("Object detection started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Frame could not be read.")
            break

        results = model(frame, verbose=False)

        annotated_frame = results[0].plot()

        cv2.imshow("Babur Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()



def detect_objects_from_frame(frame, model):
    results = model(frame, verbose=False)

    detected_objects = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        name = model.names[class_id]

        if confidence > 0.5:
            detected_objects.append(name)

    return detected_objects

if __name__ == "__main__":
    run_object_detection()
    detect_objects_from_frame()

