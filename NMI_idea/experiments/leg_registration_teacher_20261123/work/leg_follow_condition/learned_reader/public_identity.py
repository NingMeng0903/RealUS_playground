"""Load the frozen candidate probability model for newly acquired public histories."""
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

from identify import FeatureReader


class FrozenReader:
    """The contact surface and sanitized observation prefix are the only inference inputs."""

    def __init__(self, surface, model_path=None):
        path = Path(model_path) if model_path is not None else Path(__file__).parent / 'frozen/model.joblib'
        record = joblib.load(path)
        self.extractor = FeatureReader(SimpleNamespace(surface=surface))
        self.model = record['model']
        self.threshold = float(record['threshold'])

    @staticmethod
    def sanitize(legacy_frames):
        return FeatureReader.public(legacy_frames)

    def scores(self, public_frames, groups):
        """Return (frame index, echo index) pairs and target probabilities per requested group."""
        if any(frame[i] is not None for frame in public_frames for i in (3, 6, 7)):
            raise ValueError('Expected only public observations; call FrozenReader.sanitize on legacy frames')
        blocks = self.extractor.feature_sets(public_frames, groups)
        return [(indices, self.model.predict(matrix) if len(matrix) else np.empty(0))
                for matrix, indices in blocks]

    def best(self, public_frames, groups):
        """Return the best actual candidate per group, retaining its unthresholded probability."""
        decisions = []
        for ids, probability in self.scores(public_frames, groups):
            if not len(probability):
                decisions.append(None)
                continue
            k = int(probability.argmax())
            decisions.append((int(ids[k, 0]), int(ids[k, 1]), float(probability[k])))
        return decisions
