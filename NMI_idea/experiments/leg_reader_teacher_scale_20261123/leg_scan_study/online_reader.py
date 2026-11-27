"""All-contour history reader used identically by every online experiment arm."""

import numpy as np

import motion


class PublicHistoryReader(motion.Reader):
    """Rank anonymous evidence by fixed atlas relations and acquired track support."""

    threshold = .55

    def __init__(self, *args):
        super().__init__(*args)
        self._geometric = []
        self._choices = {}
        self._rankings = {}
        self._cached_count = -1

    def _new_geometry(self, frame):
        station, sector, observed, _, quality, _, posterior, _unknown, _tick = frame
        if not len(observed):
            return np.empty((0, 2)), np.empty(0)
        surface = self.model.surface
        x = np.asarray([self.absolute(station, sector, xy) for xy in observed])
        location = int(np.floor(station + .5))
        prior = surface.priors[:, location, :5]
        valid = np.isfinite(prior).all(axis=-1)
        difference = x[:, None, None] - prior[None]
        distance = np.where(valid[None], np.linalg.norm(difference, axis=-1), 120.).min(axis=1)
        target_distance = distance[:, 0]
        rival_distance = distance[:, 1:].min(axis=1)
        absolute = np.exp(-.5 * (target_distance / 17.) ** 2)
        margin = 1. / (1. + np.exp(np.clip((target_distance - rival_distance) / 5., -30., 30.)))
        relation = np.zeros((len(x), 5))
        if len(x) > 1:
            delta = x[None] - x[:, None]
            rel = prior[:, None, :] - prior[:, :, None]
            residual = delta[:, :, None, None, None] - rel[None, None]
            affinity = np.exp(-.5 * (np.linalg.norm(residual, axis=-1) / 7.5) ** 2)
            pairs = valid[:, None, :] & valid[:, :, None] & ~np.eye(5, dtype=bool)[None]
            affinity = np.where(pairs[None, None], affinity, 0.)
            ids = np.arange(len(x))
            affinity[ids, ids] = 0.
            relation = (affinity.max(axis=1).sum(axis=-1)
                        / np.maximum(pairs.sum(axis=-1)[None], 1)).max(axis=1)
        relative_margin = .5 + .5 * np.tanh(2. * (relation[:, 0] - relation[:, 1:].max(axis=1)))
        score = (.29 * absolute + .19 * margin + .18 * relation[:, 0]
                 + .095 * relative_margin + .085 * np.sqrt(posterior) + .044 * quality)
        return x, score

    def details(self, region):
        n = len(self.frames)
        if n != self._cached_count:
            self._choices.clear()
            self._rankings.clear()
            self._cached_count = n
            self._geometric.extend(self._new_geometry(frame) for frame in self.frames[len(self._geometric):])
        if region in self._choices:
            return self._choices[region]
        indices = [i for i, frame in enumerate(self.frames) if frame[5] == region and len(frame[2])]
        if not indices:
            self._choices[region] = None, 0.
            self._rankings[region] = (np.empty(0, int), np.empty(0, int), np.empty(0))
            return self._choices[region]
        lower, upper = min(self.frames[i][0] for i in indices), max(self.frames[i][0] for i in indices)
        others = [i for i, frame in enumerate(self.frames)
                  if frame[5] != region and lower - 3. <= frame[0] <= upper + 3. and len(frame[2])]
        context = indices + others
        world = np.concatenate([self._geometric[i][0] for i in context])
        quality = np.concatenate([self._geometric[i][1] for i in context])
        frame_ids = np.repeat(context, [len(self.frames[i][2]) for i in context])
        slots = np.concatenate([np.arange(len(self.frames[i][2])) for i in context])
        s = np.array([self.frames[i][0] for i in frame_ids])
        a = np.array([self.frames[i][1] for i in frame_ids])
        k = sum(len(self.frames[i][2]) for i in indices)
        delta_s = np.abs(s[:k, None] - s[None])
        delta_a = abs((a[:k, None] - a[None] + 12.) % 24. - 12.)
        different = frame_ids[:k, None] != frame_ids[None]
        supportable = different & (((delta_s >= .65) & (delta_s <= 3.))
                                   | ((delta_s <= 2.) & (delta_a >= .75)))
        distance = np.linalg.norm(world[:k, None] - world[None], axis=-1)
        affinity = np.where(supportable, np.exp(-.5 * (distance / (4. + delta_s)) ** 2), 0.)
        # Spatial agreement is independent of a candidate passing the
        # legacy target-posterior screen.
        support = (affinity * (.65 + .35 * quality[None])).max(axis=1)
        evidence = quality[:k] + .22 * support
        self._rankings[region] = frame_ids[:k], slots[:k], evidence
        best = int(np.argmax(evidence))
        choice = (int(frame_ids[best]), int(slots[best]), float(evidence[best]))
        self._choices[region] = (choice if evidence[best] >= self.threshold else None, float(support[best]))
        return self._choices[region]

    def ranked(self, region):
        self.details(region)
        return self._rankings[region]

    def select(self, region):
        if not self.history:
            return super().select(region)
        return self.details(region)[0]
