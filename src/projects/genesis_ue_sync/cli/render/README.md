# Visualization & DCC pipelines

All **Unreal Editor**, **Blender**, and **offline media** helpers for BEDLAM / dual-cam / asset conversion live here (one tree instead of scattered `scripts/ue`, `scripts/media`, `scripts/blender`).

| Subfolder | Contents |
|-----------|----------|
| `unreal/` | Editor session hooks, BEDLAM dual-cam batch, animation asset session, official sync driver |
| `blender/` | SMPL motion bundle import / FBX export profiles aligned with Unreal batch |
| `media/` | Collada→OBJ, dual-cam video compose |

Repo root for these scripts: `Path(__file__).resolve().parents[3]` from any `*.py` in `unreal/` or `blender/`.
