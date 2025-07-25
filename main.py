import cv2
from tracking.person_detection import PersonDetector
from tracking.person_tracking import PersonTracker
from utils.data_manager import DataBuffer
from utils.profiler import PipelineProfiler
import time
import psutil
import os

if __name__ == "__main__":
    cap = cv2.VideoCapture("../v_0/input/3c.mp4")  # Use video file path if needed
    detector = PersonDetector()
    tracker = PersonTracker()
    buffer = DataBuffer()
    profiler = PipelineProfiler()

    frame_id = 0
    profiler.start_frame_processing()
    while cap.isOpened() and frame_id < 500:
        ret, frame = cap.read()
        if not ret:
            break

        profiler.start("Detection")
        detector.detect_and_store(frame, frame_id, buffer)
        profiler.stop("Detection")

        profiler.start("Tracking")
        tracker.update_tracks(frame_id, buffer, profiler)
        profiler.stop("Tracking")

        tracks = buffer.get_tracks(frame_id)
        for t in tracks:
            x1, y1, x2, y2, track_id = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    profiler.end_frame_processing(frame_id)
    print(profiler.get_summary())

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"\n--- Memory Usage ---")
    print(f"RSS (Resident Set Size): {memory_info.rss / (1024 * 1024):.2f} MB")
    print(f"VMS (Virtual Memory Size): {memory_info.vms / (1024 * 1024):.2f} MB")
    print(f"--- End of Memory Usage ---")

    cap.release()
    cv2.destroyAllWindows()
