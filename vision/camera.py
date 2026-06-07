import cv2


def open_camera(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Camera could not be opened.")
        return

    print("Camera started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Frame could not be read.")
            break

        cv2.imshow("Babur Vision", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if cv2.waitKey(1) & 0xFF == ord("s"):
            cv2.imwrite("data/camera_snapshot.jpg", frame)
            print("Snapshot saved.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    open_camera()