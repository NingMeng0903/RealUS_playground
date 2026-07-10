# Human motion validation

English-only notes (project convention).

## Capsule vs SMPL FK

The shape capsule URDF uses **three continuous joints** (`ex`, `ey`, `ez`) per body segment to approximate each SMPL **single joint axis-angle**. Even with a correct Genesis DoF layout and zero readback error, **capsule collision geometry will not coincide limb-for-limb** with the skinned SMPL mesh; expect larger errors at shoulders, hips, and knees.

**Dual track (Genesis only):** a parallel **MJCF** proxy with `freejoint` + per-bone `ball` joints is generated under `outputs/.../smpl_proxy_mjcf/` when `prepare_smpl_capsule_runtime_asset(..., genesis_proxy="mjcf")` is used. **Unreal / UE** continues to consume the existing **URDF** cache under `smpl_proxy_urdf/`; the UE bridge is unchanged.

## Tools

- `run_capsule_frame_audit` in `capsule_frame_audit.py`: one-frame JSON report (packed `q`, readback after `step`, root translation sources, per-link **position and orientation** error vs SMPL FK, floating-base layout check). Pass `genesis_proxy="mjcf"` for the MJCF asset.
- CLI: `python -m projects.genesis_ue_sync.cli.pipeline.support_motion.audit_capsule_smpl_single_frame` (add `--genesis-human-proxy mjcf` for MJCF).
- DoF packing vs Genesis: `verify_capsule_smpl_alignment.py` (`--discover-cache` for URDF, `--discover-mjcf` for MJCF smoke when a cached MJCF exists).

## Motion manifests

`write_motion_manifest` attaches `metrics["physical_refit"]` when `PhysicalRefitDiagnostics` is present. Refit diagnostics include `loss_weights` (Hamiltonian scalars), merged `contact_mask_normalized` (`CONTACT_KEYS`), and MVP notes distinguishing offline objectives from PD playback.
