# 17:23 手柄 hybrid 退出：不是 QP 无解

四场都进入了 `TRACK_HYBRID` 并跑过力控。控制器上的 `qpfeasible` 观感来自随后的模式退出，不是安装瞬间无解。

| 场次 | 成功提交 | 时长 | 日志中的退出 |
| --- | ---: | ---: | --- |
| 172312 | 710 | 22.8 s | 下一拍 Fz 越 6 N（末拍 5.41 N，已在 −15 mm/s 回缩） |
| 172340 | 19 | 0.63 s | 再接触后 5.18 N 回缩，下一拍越 6 N |
| 172345 | 98 | 3.48 s | `control_interval_expired (0.055 s)`，力约 4.2 N |
| 172358 | 509 | 14.6 s | `ESTOP pad_native_timeout`（native coast 被当成急停） |

全程 `image_invalid`：`nearfield_echo_feature_missing_or_incompatible`。视觉和 COP 没有参与已发送命令。

## 原因

导轨 `commit_reservation(acknowledge=True, timeout_s=0.04)` 把控制周期从约 5 ms 拖到 37–40 ms（偶发 54–55 ms）。力环按这个实际周期积分，Z 命令几乎总是打满 ±15 mm/s，Fz 在 2.6–5.7 N 之间来回。单帧 6 N 监督在已经回缩时仍退出 hybrid，探头停在高载荷。55 ms 超过当时的 50 ms 命令寿命。一次 native 20 ms 唤醒丢失被手柄路径写成 ESTOP；扫描路径本来会 coast。

## 修复

- 手柄导轨提交改回与扫描相同的入队成功，不再等待 50 Hz worker 写完。
- 单次 `native_timeout_coast` 撤销未发送候选并继续；进程退出或请求过期的锁定 `native_timeout` 仍急停。
- 6 N 短时过冲继续 4 N 回缩；持续 0.15 s 或达到 8 N 才停机。
- `command_max_age_s` 改为 0.08，覆盖偶发长周期。

只改手柄路径。部署后必须重启控制器和手柄，并单独重启带 `nearfield_column_echo_v1` 的特征进程，否则图像仍会是 `image_invalid`。
