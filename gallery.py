import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class PersonGallery:
    """
    Stores feature vectors for known IDs and matches new features against them.
    """
    def __init__(self, threshold=0.7, max_missed=100):
        self.threshold = threshold
        self.max_missed = max_missed
        self.next_id = 0
        self.features = {}       # id -> list of feature vectors
        self.last_seen = {}      # id -> last frame index

    def match(self, feat, frame_idx):
        """
        Given a single feature vector, return matched_id or None.
        """
        if not self.features:
            return None

        # Compute cosine similarity between feat and the mean feature of each id
        gallery_ids = []
        gallery_feats = []
        for pid, feats in self.features.items():
            mean_feat = np.mean(feats, axis=0)
            gallery_ids.append(pid)
            gallery_feats.append(mean_feat)

        sims = cosine_similarity([feat], gallery_feats)[0]
        best_idx = np.argmax(sims)
        if sims[best_idx] >= self.threshold:
            matched_id = gallery_ids[best_idx]
            # update gallery and last_seen
            self.features[matched_id].append(feat)
            self.last_seen[matched_id] = frame_idx
            return matched_id
        return None

    def register(self, feat, frame_idx):
        pid = self.next_id
        self.next_id += 1
        self.features[pid] = [feat]
        self.last_seen[pid] = frame_idx
        return pid

    def prune(self, frame_idx):
        """
        Remove IDs not seen for more than max_missed frames.
        """
        to_remove = [pid for pid, last in self.last_seen.items()
                     if frame_idx - last > self.max_missed]
        for pid in to_remove:
            del self.features[pid]
            del self.last_seen[pid]