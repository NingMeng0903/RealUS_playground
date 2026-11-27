"""Reader variant that ranks tangents by common-parent integrated evidence."""
from operator import itemgetter

from .baselines import Full_AGPS_Reader
from .dynamic_factor_graph import _Proposal, _contributor_masks


class EvidenceRankedReader(Full_AGPS_Reader):
    """Change only the ranking that selects paths for v6 geometry refinement."""

    def _propose(self, frame, design, parents, seeds, rho):
        states = [seed for _, seed in seeds]
        predictions = self._predict_many(frame, design, states)
        solved = self._belief_rhs_many(states, predictions, design.cols)
        limit = self.max_branches + 2
        shortlist = {}
        for j, ((i, seed), pred, rhs) in enumerate(zip(seeds, predictions, solved, strict=True)):
            parent = parents[i]
            for a in self._associate(frame, design, seed, pred, solved=rhs):
                if seed is not parent and not len(a.echoes):
                    # No geometry supports moving a NULL path off its parent.
                    continue
                masks = _contributor_masks(a.groups)
                observation = frame.echoes[a.echoes, :2].ravel()
                mean, data, evidence, objective = self._local_factor(
                    parent, seed.mean, pred, design, a, observation, rho, rhs)
                rank = (-(parent.log_weight + evidence), i, masks, j)
                key = (i, masks)
                if key in shortlist:
                    if rank >= shortlist[key][0]:
                        continue
                elif len(shortlist) == limit:
                    worst = max(shortlist, key=lambda k: shortlist[k][0])
                    if rank >= shortlist[worst][0]:
                        continue
                    del shortlist[worst]
                # Seeds are alternate tangents, not extra mass. Equal masks
                # from distinct parents still represent different histories.
                shortlist[key] = (rank, _Proposal(i, a, masks, observation, seed.mean, pred,
                                                  mean, data, evidence, objective))

        ranked = sorted(shortlist.values(), key=itemgetter(0))
        shortlist.clear()
        proposals = []
        for rank, proposal in ranked:
            proposals.append(proposal)
            # Preserve v6's inclusion of the first path that crosses the gap,
            # and its minimum of three proposals when available.
            if len(proposals) >= 3 and rank[0] > ranked[0][0][0] + 6.:
                break
        del ranked
        self._refine(frame, design, parents, proposals, rho)
        return proposals
