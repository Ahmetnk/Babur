import cv2
from ultralytics import YOLO
import torch

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

def detect_objects_from_results(results, model, confidence_threshold: float = 0.5):
    detected_objects = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        name = model.names[class_id]

        print(f"{name} | confidence = {confidence:.2f}")

        if confidence > confidence_threshold:
            detected_objects.append(name)

    return list(set(detected_objects))


def run_object_detection(camera_index: int = 0):
    model = YOLO("yolo11n.pt")

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

        results = model(frame,device=0 ,verbose=False)

        detected_objects = detect_objects_from_results(results, model)
        print("Detected objects:", detected_objects)

        annotated_frame = results[0].plot()

        cv2.imshow("Babur Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_object_detection()