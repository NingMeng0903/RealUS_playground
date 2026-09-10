"""Opt-in final publication adapter for a generic Cartesian-certified candidate.

The certified native payload is immutable: a reservation clamp or later Python
correction requires another native solve. Legacy modes retain their own sender.
No hardware connection is opened here; transports are supplied by the runner.
"""
from __future__ import annotations

import time
import numpy as np

from peirastic.contact_qp.execution import DeviceState, FinalCommand


def publish_contact_candidate(outer, inner, step, rail, *, native_q_send, native_qdot,
                              send_arm, publication_gate, now=time.monotonic):
    """Reserve, review the last payload, publish per device, commit confirmed facts.

    `publication_gate` raises on stop/stale feedback; it runs before each send.
    SDK exceptions mean unknown exposure. Rail commit False means no new worker
    exposure because the required reservation interface owns that fact.
    """
    coordinator = outer.execution
    publication = None
    active_device = None
    try:
        publication_gate()
        if rail is None or not getattr(rail, "enabled", False):
            raise ValueError("contact_qp requires the arm/rail execution adapter")
        if not all(callable(getattr(rail, method, None)) for method in (
                "reserve_target_m", "commit_reservation", "abort_reservation", "fence_publications")):
            raise ValueError("rail lacks versioned reservation protocol")
        q = np.asarray(step.q_send, dtype=float).copy()
        qdot = np.asarray(step.qdot, dtype=float).copy()
        if (q.shape != (8,) or qdot.shape != (8,) or not np.isfinite(q).all() or not np.isfinite(qdot).all()
                or not np.array_equal(q, native_q_send) or not np.array_equal(qdot, native_qdot)):
            raise ValueError("final payload differs from native mechanical certificate")
        cert = outer.pending_result.hard_constraints
        if (not bool(step.cartesian_constraints_valid)
                or step.cartesian_constraints_sequence != cert.sequence
                or step.cartesian_constraints_stop_epoch != cert.stop_epoch
                or coordinator.stop_epoch != cert.stop_epoch):
            raise ValueError("native generic certificate missing or belongs to another candidate")
        if not rail.reserve_target_m(float(q[0]), v_ff_m_s=float(qdot[0]),
                                     candidate_sequence=cert.sequence, stop_epoch=cert.stop_epoch,
                                     valid_until_s=cert.valid_until_s):
            raise ValueError("rail reservation rejected")
        reserved = rail.reserved_publication
        if (reserved is None or reserved["target_m"] != float(q[0])
                or reserved["v_ff"] != float(qdot[0]) or reserved["candidate_sequence"] != cert.sequence):
            raise ValueError("final rail reservation changed native payload")
        rotation = np.asarray(outer.pending_rotation_base_tcp, dtype=float)
        rotate = np.zeros((6, 6)); rotate[:3, :3] = rotate[3:, 3:] = rotation.T
        command = FinalCommand(int(cert.sequence), int(cert.stop_epoch), np.rad2deg(q[1:]), float(q[0]),
            rotate @ np.asarray(step.v_tcp_commanded), rotate @ np.asarray(step.v_tcp_estimated),
            rotate @ np.asarray(step.arm_model_twist), rotate @ np.asarray(step.rail_model_twist), True)
        publication = coordinator.review(command, cert, now_s=now(), h=outer.pending_basis,
            alpha=outer.pending_result.alpha, dt_s=outer.pending_dt,
            wrench_environment=outer.pending_wrench_environment)
        publication_gate()
        coordinator.before_send(publication, "arm", now_s=now(), command=command)
        active_device = "arm"
        step.arm_send_mono_ns = int(now()*1e9)
        send_arm(command.arm_payload.copy())
        coordinator.device_result(publication, "arm", DeviceState.SENT, now_s=now())
        active_device = None
        publication_gate()
        coordinator.before_send(publication, "rail", now_s=now(), command=command)
        active_device = "rail"
        if not rail.commit_reservation(candidate_sequence=cert.sequence, stop_epoch=cert.stop_epoch):
            coordinator.device_result(publication, "rail", DeviceState.REJECTED, now_s=now())
            active_device = None
            raise ValueError("PARTIAL_ARM:rail_commit_failed")
        coordinator.device_result(publication, "rail", DeviceState.SENT, now_s=now())
        active_device = None
        # The final outer commit repeats time/energy/fact checks. Only after a
        # complete publication may native history learn the published action.
        if inner.commit_publication(qdot) is not True:
            raise ValueError("published_command_native_history_confirmation_failed")
        outer.publication_commit(publication, now_s=now())
        if not publication.committed:
            raise ValueError("publication_complete_but_reference_commit_refused")
        return command
    except Exception as exc:
        if publication is not None:
            if active_device is not None:
                coordinator.device_result(publication, active_device, DeviceState.UNKNOWN, now_s=now())
            if not publication.committed:
                coordinator.abort(publication, str(exc))
        abort_rail = getattr(rail, "abort_reservation", None)
        if callable(abort_rail):
            abort_rail()
        inner.abort_publication()
        outer.publication_abort(str(exc))
        raise
