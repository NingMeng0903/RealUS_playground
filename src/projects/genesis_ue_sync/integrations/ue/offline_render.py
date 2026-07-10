"""Unreal offline rendering boundary (MRQ / Sequencer / curated pipelines).

Realtime UE captures flow through `integrations/ue/cpp_plugin/AmongUsRealtimeCapture` plus
`cli/render/unreal/amongus_ue_tcp_camera_mux.py`.

Offline dataset renders MUST stay decoupled from realtime loops:

- Build canonical JSONL via `AMONGUS_GENESIS_CANONICAL_STATE_JSONL` during simulation.
- Replay states inside UE (`cli/render/unreal/ue_replay_canonical_state_log.py`) or regenerate MRQ jobs via
  `cli/render/unreal/ue_bedlam_dual_cam_batch.py`.

Keeping offline tooling isolated avoids coupling latency-sensitive subscribers with Movie Render Queue disk IO."""

__all__: list[str] = []
