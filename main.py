import cv2
from person_detection import PersonDetector
from person_tracking import PersonTracker
from data_manager import DataBuffer
import time

if __name__ == "__main__":
    cap = cv2.VideoCapture("../v_0/input/3c.mp4")  # Use video file path if needed
    detector = PersonDetector()
    tracker = PersonTracker()
    buffer = DataBuffer()

    start_time = time.time()
    frame_id = 0
    while cap.isOpened() and frame_id < 600:
        ret, frame = cap.read()
        if not ret:
            break

        detector.detect_and_store(frame, frame_id, buffer)
        tracker.update_tracks(frame_id, buffer)

        tracks = buffer.get_tracks(frame_id)
        for t in tracks:
            x1, y1, x2, y2, track_id = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    end_time = time.time()
    print(f"Processed {frame_id} frames in {end_time - start_time:.2f} seconds.\n FPS: {frame_id / (end_time - start_time):.2f}")
    cap.release()
    cv2.destroyAllWindows()
