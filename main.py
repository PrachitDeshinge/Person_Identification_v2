import config
import cv2
from tracking.person_detection import PersonDetector
from tracking.person_tracking import PersonTracker
from silhouette.u2net_silhouette import U2NetSilhouetteGenerator
from utils.data_manager import DataBuffer
from utils.profiler import PipelineProfiler
import time
import psutil
import os

def main():
    # Input source
    cap = cv2.VideoCapture(config.INPUT_VIDEO)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open input video: {config.INPUT_VIDEO}")
    
    # --- Video Writer Setup ---
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    tracking_writer = None
    if config.OUTPUT_TRACKING_VIDEO:
        tracking_writer = cv2.VideoWriter(config.OUTPUT_TRACKING_VIDEO, fourcc, fps, (width, height))
        if tracking_writer.isOpened():
            print(f"📹 Saving tracking output to: {config.OUTPUT_TRACKING_VIDEO}")
        else:
            print(f"[ERROR] Failed to open tracking video writer.")
            tracking_writer = None

    silhouette_writer = None
    if config.OUTPUT_SILHOUETTE_VIDEO:
        # Silhouette video is grayscale, so it has only one channel, but we save it in BGR format
        silhouette_writer = cv2.VideoWriter(config.OUTPUT_SILHOUETTE_VIDEO, fourcc, fps, (width, height))
        if silhouette_writer.isOpened():
            print(f"👤 Saving silhouette output to: {config.OUTPUT_SILHOUETTE_VIDEO}")
        else:
            print(f"[ERROR] Failed to open silhouette video writer.")
            silhouette_writer = None

    detector = PersonDetector()
    tracker = PersonTracker()
    silhouette_generator = U2NetSilhouetteGenerator()
    buffer = DataBuffer()
    profiler = PipelineProfiler()

    frame_id = 0
    profiler.start_frame_processing()
    max_frames = config.MAX_FRAMES if config.MAX_FRAMES and config.MAX_FRAMES > 0 else float('inf')
    while cap.isOpened() and frame_id < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        profiler.start("Detection")
        detector.detect_and_store(frame, frame_id, buffer)
        profiler.stop("Detection")

        profiler.start("Tracking")
        tracker.update_tracks(frame, frame_id, buffer, profiler)
        profiler.stop("Tracking")

        tracks = buffer.get_tracks(frame_id)
        
        profiler.start("Silhouette")
        silhouette_mask = silhouette_generator.generate_silhouette(frame, tracks)
        profiler.stop("Silhouette")

        # --- Visualization ---
        
        # Draw tracking info on the original frame
        tracking_display = frame.copy()
        for t in tracks:
            x1, y1, x2, y2, track_id = map(int, t)
            cv2.rectangle(tracking_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(tracking_display, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Create a visual representation of the silhouette
        silhouette_display = cv2.cvtColor(silhouette_mask, cv2.COLOR_GRAY2BGR)

        # Write frames to output videos
        if tracking_writer:
            tracking_writer.write(tracking_display)
        if silhouette_writer:
            silhouette_writer.write(silhouette_display)

        # Display windows if not in headless mode
        if not config.HEADLESS:
            cv2.imshow("Tracking", tracking_display)
            cv2.imshow("Silhouette", silhouette_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            if frame_id % 100 == 0:
                print(f"📊 Processed frame {frame_id}, found {len(tracks)} tracks")

        buffer.cleanup(frame_id)
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
    if tracking_writer:
        tracking_writer.release()
    if silhouette_writer:
        silhouette_writer.release()

    if not config.HEADLESS:
        cv2.destroyAllWindows()
    
    print(f"🎬 Processing completed - {frame_id} frames processed")
    if config.OUTPUT_TRACKING_VIDEO:
        print(f"✅ Tracking video saved to: {config.OUTPUT_TRACKING_VIDEO}")
    if config.OUTPUT_SILHOUETTE_VIDEO:
        print(f"✅ Silhouette video saved to: {config.OUTPUT_SILHOUETTE_VIDEO}")
        
    if config.HEADLESS:
        print("🖥️  Ran in headless mode (no GUI)")
    else:
        print("🖼️  Ran with GUI display")

if __name__ == "__main__":
    main()