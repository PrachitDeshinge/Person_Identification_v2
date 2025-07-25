import torch
from torchvision import transforms
from PIL import Image
from models.vit_model import CompleteVisionTransformer

class ReIDManager:
    def __init__(self, similarity_threshold=0.7, lost_track_buffer=30):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._load_model()
        self.transform = self._get_transform()
        
        self.similarity_threshold = similarity_threshold
        self.lost_track_buffer = lost_track_buffer

        self.active_gallery = {}
        self.lost_gallery = {}

    def _load_model(self):
        model = CompleteVisionTransformer()
        checkpoint = torch.load('../weights/transformer_120.pth', map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        model.to(self.device)
        model.eval()
        print("[SUCCESS] Re-ID model loaded successfully!")
        return model

    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def extract_feature(self, crop):
        image = Image.fromarray(crop).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.model(image_tensor)
        return feature

    def match_feature(self, new_feature):
        best_match_id = None
        max_similarity = -1

        # Check against lost tracks
        for lost_id, (lost_feature, _) in list(self.lost_gallery.items()):
            similarity = torch.nn.functional.cosine_similarity(new_feature, lost_feature).item()
            if similarity > self.similarity_threshold and similarity > max_similarity:
                max_similarity = similarity
                best_match_id = lost_id
        
        return best_match_id

    def register_feature(self, final_id, feature):
        if final_id in self.lost_gallery:
            del self.lost_gallery[final_id]
        self.active_gallery[final_id] = feature

    def handle_lost_tracks(self, lost_ids, frame_id):
        for track_id in lost_ids:
            if track_id in self.active_gallery:
                feature = self.active_gallery.pop(track_id)
                self.lost_gallery[track_id] = (feature, frame_id)

    def cleanup_lost_gallery(self, frame_id):
        # Remove tracks that have been lost for too long
        lost_ids_to_remove = []
        for track_id, (_, lost_frame_id) in self.lost_gallery.items():
            if frame_id - lost_frame_id > self.lost_track_buffer:
                lost_ids_to_remove.append(track_id)
        
        for track_id in lost_ids_to_remove:
            del self.lost_gallery[track_id]
