from __future__ import annotations

from typing import Any

import numpy as np


def read_virtual_contact_wrench(
    runtime: Any,
    robot_name: str,
    *,
    link_name: str | None = None,
) -> np.ndarray:
    return np.asarray(
        runtime.get_wrench(robot_name, source="sim_contact", link_name=link_name),
        dtype=np.float32,
    ).reshape(6)


def read_virtual_contact_force_world(
    runtime: Any,
    robot_name: str,
    *,
    link_name: str,
) -> np.ndarray:
    force = np.asarray(runtime.get_link_contact_force(robot_name, link_name), dtype=np.float32).reshape(3)
    if float(np.linalg.norm(force)) > 1e-9:
        return force
    contacts = runtime.get_entity_contacts(robot_name, exclude_self_contact=False)
    total = np.zeros(3, dtype=np.float32)
    for contact in contacts:
        if contact.get("link_a_name") == link_name:
            total += np.asarray(contact.get("force_a", np.zeros(3)), dtype=np.float32).reshape(3)
        if contact.get("link_b_name") == link_name:
            total += np.asarray(contact.get("force_b", np.zeros(3)), dtype=np.float32).reshape(3)
    return total
