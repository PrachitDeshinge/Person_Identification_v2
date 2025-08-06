import config

class DataBuffer:
    def __init__(self):
        self.frames = {}
        self.detections = {}
        self.tracks = {}
        self.max_buffer_size = config.REID_LOST_TRACK_BUFFER + 10 # Keep a little extra

    def store_detections(self, frame_id, frame, detections):
        self.frames[frame_id] = frame
        self.detections[frame_id] = detections

    def get_detections(self, frame_id):
        return self.detections.get(frame_id, [])

    def get_frame(self, frame_id):
        return self.frames.get(frame_id)

    def store_tracks(self, frame_id, tracks):
        self.tracks[frame_id] = tracks

    def get_tracks(self, frame_id):
        return self.tracks.get(frame_id, [])

    def cleanup(self, current_frame_id):
        if current_frame_id > self.max_buffer_size:
            cutoff_frame_id = current_frame_id - self.max_buffer_size
            for frame_id_to_remove in list(self.frames.keys()):
                if frame_id_to_remove < cutoff_frame_id:
                    del self.frames[frame_id_to_remove]
                    if frame_id_to_remove in self.detections:
                        del self.detections[frame_id_to_remove]
                    if frame_id_to_remove in self.tracks:
                        del self.tracks[frame_id_to_remove]
