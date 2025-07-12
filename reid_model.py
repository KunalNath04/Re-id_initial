import torch
# Support both torchreid layouts:
from torchreid.reid.utils.feature_extractor import FeatureExtractor


class ReidModel:
    """
    Loads a pre-trained OSNet model (via torchreid) and provides a feature-extraction interface.
    """
    def __init__(self, model_name='osnet_x1_0', device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        # FeatureExtractor wraps preprocessing and model forward
        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=None,      # uses default weights
            device=self.device
        )

    def extract(self, images):
        """
        Given a list of RGB numpy arrays (HxWxC), returns a numpy array of shape (N, D)
        where D is the embedding dimension.
        """
        # FeatureExtractor expects list of BGR or PIL images? It uses OpenCV by default.
        feats = self.extractor(images)
        # feats is torch.Tensor on device
        return feats.cpu().numpy()