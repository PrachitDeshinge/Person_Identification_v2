class DataBuffer:
    def __init__(self):
        self.frames = {}
        self.detections = {}
        self.tracks = {}

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
