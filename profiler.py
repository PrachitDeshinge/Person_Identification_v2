import time
from collections import defaultdict

class PipelineProfiler:
    def __init__(self):
        self.timings = defaultdict(float)
        self.counts = defaultdict(int)
        self.start_times = {}
        self.total_start_time = None
        self.total_frames = 0
        self.total_end_time = None

    def start_frame_processing(self):
        if self.total_start_time is None:
            self.total_start_time = time.time()

    def end_frame_processing(self, frame_count):
        self.total_frames = frame_count
        self.total_end_time = time.time()

    def start(self, stage_name):
        self.start_times[stage_name] = time.time()

    def stop(self, stage_name):
        if stage_name in self.start_times:
            duration = time.time() - self.start_times[stage_name]
            self.timings[stage_name] += duration
            self.counts[stage_name] += 1
            del self.start_times[stage_name]

    def get_summary(self):
        if self.total_start_time is None or self.total_end_time is None:
            return "Profiler was not used correctly (start/end frame processing not called)."

        total_duration = self.total_end_time - self.total_start_time
        fps = self.total_frames / total_duration if total_duration > 0 else 0

        summary = []
        summary.append("--- Pipeline Profiling Summary ---")
        summary.append(f"Processed {self.total_frames} frames in {total_duration:.2f} seconds. Overall FPS: {fps:.2f}")
        summary.append("\n--- Stage-wise Analysis ---")

        # Sort stages by total time descending
        sorted_stages = sorted(self.timings.items(), key=lambda item: item[1], reverse=True)

        for name, total_time in sorted_stages:
            count = self.counts[name]
            avg_time_ms = (total_time / count) * 1000 if count > 0 else 0
            total_time_percent = (total_time / total_duration) * 100 if total_duration > 0 else 0
            summary.append(f"- Stage: {name:<20} | Total: {total_time:7.2f}s ({total_time_percent:5.2f}%) | Avg: {avg_time_ms:7.2f} ms/call | Calls: {count}")
        
        summary.append("\n--- End of Summary ---")
        return "\n".join(summary)
