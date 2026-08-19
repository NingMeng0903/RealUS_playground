"""WBC velocity-IK solver interfaces (Escande slack QP)."""

from .branch_barrier import BranchBarrierBuilder, BranchBarrierConfig, latch_q_star_signs
from .constraint_mgr import VelocityBoxConstraints
from .qp_builder import QpConfig, QpIkController
from .sigma_setbased import SigmaSetBasedConfig, SigmaSetBasedTracker

__all__ = [
    "BranchBarrierBuilder",
    "BranchBarrierConfig",
    "QpConfig",
    "QpIkController",
    "SigmaSetBasedConfig",
    "SigmaSetBasedTracker",
    "VelocityBoxConstraints",
    "latch_q_star_signs",
]
