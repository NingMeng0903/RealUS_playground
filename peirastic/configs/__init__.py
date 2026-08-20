"""Peirastic YAML roots. Apps must not point at rm75_control/configs."""

from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_CONTROLLER_YAML = CONFIG_DIR / "controller.yaml"
DEFAULT_FORCE_YAML = CONFIG_DIR / "force.yaml"

__all__ = ["CONFIG_DIR", "DEFAULT_CONTROLLER_YAML", "DEFAULT_FORCE_YAML"]
