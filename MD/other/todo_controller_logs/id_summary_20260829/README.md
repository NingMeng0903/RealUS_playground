# RealMan 2026-08-29 动力学辨识汇总

空中植物是 `vel_ff_vz → vz_achieved_tool` 的 FOPDT，不是力口 `Ya = v / F_ext`。
能写入控制器的中心数是 **T0 = 35 ms、Tp = 12 ms**。3 Hz 群延迟不要当 T0。
`system_delay_s = 0.055` 从未测过。接触刚度是夹具，不是组织。

本目录只做汇总。原始 `id_air_*` / `id_fit_*` 未改。

## 先看这些数

| 量 | 值 | 怎么用 |
|---|---|---|
| T0（chirp 纯延迟） | **35 ms**，三段 8/15/25 mm/s 完全相同 | 写 `safety_shield.plant.t0_s`。已写入。 |
| Tp（一阶时间常数） | **12 ms**（三段中位数） | 写 `safety_shield.plant.tp_s`。已写入。单段 Tp 随 `chirp_fits.csv` 一起丢了。 |
| Td 区间 | **35–41 ms**（T0 + `feedback_age` p95 6.3 ms） | 证书把延迟当区间，不当点。p95 已超过 1 拍（5 ms）。 |
| 线性压速标签 | **≤ 40 mm/s** | +20 mm/s 峰加速已约 2.4 m/s²。首触仍用 8–12，不要用 40，力轴空中禁 80。 |
| 阶跃 xcorr 延迟 | 大多 **30–45 ms**，对齐 35 | 见 [S05](S05_delay_vs_speed.png)。−6 mm/s 有一个 0 ms 野点。 |
| 80 mm/s 加速 | 约 **4.1–4.3 m/s²**，回零上升 105–185 ms | 力轴不要当空气/辨识压速用 80。 |
| 稳态速度增益 Kp | ≥20 mm/s 大约 **0.96–1.01** | 旧 yaml 0.892 偏低。低速（≤12）散。 |
| 夹具 Ke | **1420 N/m**（8 mm/s）和 **2199 / 2522 N/m**（3 mm/s） | 两块垫，不是组织。yaml `ke_cap_ub=2000` 卡在中间。 |
| 没测到的 | 力环 F→v FRF；单组织 Ke；停-反 `Δx_b^ub` | 不要开 CDYOB，不要把 shield 标 certified。 |

模型：

```
Gv(s) = K · exp(-T0 s) / (Tp s + 1)
```

拟合在 [`peirastic/apps/identify_plant.py`](../../../peirastic/apps/identify_plant.py) 的 `_fit_fopdt`：先互相关锁通信延迟，再在 `Tp ∈ [8, 40] ms` 上扫，增益夹在 `[0.7, 1.3]`。相关延迟不和 Tp 对换。

## 文件从哪来

| 角色 | 路径 | 还在盘上的 |
|---|---|---|
| **规范硬件空中** | [`../id_air_20260829_202437/`](../id_air_20260829_202437/) | `01`–`07` png/svg，`jitter.json`，`tdpa_shadow.json`。**42414** 行，T0 spread 0，线性标签 40 mm/s。 |
| 同日较早硬件 | [`../id_air_20260829_200549/`](../id_air_20260829_200549/) | 同样有图和两个 JSON。**45567** 行，T0 spread 5 ms，线性标签 30 mm/s。**不是**写入 yaml 的那次。 |
| 合成（非硬件） | [`../id_air_20260829_synthetic/`](../id_air_20260829_synthetic/) | 分析器干跑。`write_yaml_allowed=false`。 |
| 打包拟合 | [`../id_fit_20260829/`](../id_fit_20260829/) | [`identification.json`](../id_fit_20260829/identification.json)，[`id_reference.log`](../id_fit_20260829/id_reference.log)，`08_contact_press`。 |
| 两次开环压 | `id_tdpa_20260829_210139`、`212852` | **目录已不在**。Ke / 符号只留在 `identification.json`。 |

当天分析器还写过、现在已经删掉的：

- 各空中目录的 `air_campaign.csv` / Window A 日志、`edges.csv`、`chirp_fits.csv`
- 两次压的 `tdpa_press.csv`
- `id_fit_20260829/contact_ke.csv`、`controller_ref.csv`

202437 跑完时分析器打印过 `edges=51 chirps=3 linear≤40.0 mm/s T0_spread=0.0 ms`。

**本目录怎么补表**

- 标量来自还在的 JSON。
- 202437 的 chirp：三段 T0 都是 35 ms（spread 0）；Tp 只有中位数 12 ms；3 Hz 群延迟来自 `identification.json`；**单段 gain 补不回来**。
- 阶跃散点从 matplotlib SVG 的 marker 反解（`edges_from_svg.csv`）。这不是原始 `edges.csv`：没有 `t_edge_s`，04 图分不清正负号。
- 200549 的分段 chirp 和一张拼好的阶跃表，来自当时还在的 CSV 的同期重算（80 ms 预窗），不是今天新拟合。

## 本目录清单

**表**

- [`plant_summary.csv`](plant_summary.csv) — 三次 campaign 的 T0/Tp/抖动对照
- [`chirp_fits.csv`](chirp_fits.csv) — 8/15/25 mm/s FOPDT + 3 Hz 群延迟
- [`edges_202437_step_delay.csv`](edges_202437_step_delay.csv) — 规范次 ±阶跃延迟（从 02 图还原）
- [`edges_from_svg.csv`](edges_from_svg.csv) — 01–04 全部还原点（含 `suspect`）
- [`edges_200549_reconstructed.csv`](edges_200549_reconstructed.csv) — 较早次 delay/rise/accel/Kp
- [`contact_ke.csv`](contact_ke.csv) — 两次夹具压
- [`tdpa_air_shadow.csv`](tdpa_air_shadow.csv) — 空气 TDPA 记账（不是刚性压符号检查）
- [`yaml_vs_id.csv`](yaml_vs_id.csv) — 辨识值对照 2026-09-02 的 [`force.yaml`](../../../peirastic/configs/force.yaml)

**新图（本目录根下）**

- [`S01_plant_headline.png`](S01_plant_headline.png) — T0/Tp/抖动 vs yaml 占位
- [`S02_group_delay_vs_t0.png`](S02_group_delay_vs_t0.png) — 为什么 65/49/20 ms 不能当 T0
- [`S03_jitter_compare.png`](S03_jitter_compare.png) — 三次 campaign 抖动
- [`S04_contact_ke.png`](S04_contact_ke.png) — 夹具 Ke（端点斜率；原始 F–x 轨迹已删）
- [`S05_delay_vs_speed.png`](S05_delay_vs_speed.png) — 阶跃延迟
- [`S06_accel_and_rise.png`](S06_accel_and_rise.png) — 加速响应
- [`S07_kp_vs_speed.png`](S07_kp_vs_speed.png) — 稳态增益
- [`bode_Gv_velocity.png`](bode_Gv_velocity.png) — velocity-tracking Bode; x-axis is frequency (Hz), dashed line at 8 Hz

每张也有同名 `.svg`。

**当天原图（只复制，未改）** — [`original_campaign_plots/`](original_campaign_plots/)

`01_delay_vs_accel` … `07_jitter` 来自 202437；`08_contact_press` 来自 `id_fit`。Bode 原图仍是看频率形状的最好入口。

## 空中植物（规范次 202437）

激励：`AIR_STEPS_MM_S = 4…80`，保持 1.0 s，休息 0.5 s；三段 60 s chirp，0.2–8 Hz，幅值 8/15/25 mm/s。力环关着。配的是命令速度和工具 Z 实测速度，不是力误差。

### Chirp FOPDT

| 幅值 mm/s | T0 ms | Tp ms | K | 测到的 3 Hz 群延迟 ms | FOPDT 在 3 Hz 的群延迟 ms |
|---|---:|---:|---:|---:|---:|
| 8 | 35 | 12（中位） | （丢了） | 64.5 | 46.4 |
| 15 | 35 | 12（中位） | （丢了） | 49.5 | 46.4 |
| 25 | 35 | 12（中位） | （丢了） | 20.2 | 46.4 |

FOPDT 在 3 Hz 的群延迟是 `T0 + Tp / (1 + (ω Tp)²)`。只有 15 mm/s 的测量接近这个数。65/49/20 对不上同一个二参数模型，所以 `do_not_use_group_delay_as_t0: true`。

较早的 200549 分段还在（同期分析）：8 mm/s → 35/22 ms、K=0.869；15 → 30/14、0.947；25 → 30/12、0.966。那次 T0 spread 5 ms，没有写成 yaml。

### 阶跃延迟（202437，从 02 图还原）

| mm/s | +cmd ms | −cmd ms |
|---:|---:|---:|
| 4 | 35 | 45 |
| 6 | 40 | **0**（野点） |
| 8 | 40 | 40 |
| 10 | 45 | 40 |
| 12 | 35 | 35 |
| 15 | 30 | 35 |
| 20 | 40 | 35 |
| 25 | 45 | 30 |
| 30 | 20 | 45 |
| 40 | 40 | 40 |
| 60 | 45 | 35 |
| 80 | 40 | 40 |

线性区里多数点在 30–45 ms。延迟不随速度单调变长；80 mm/s 仍然约 40 ms，坏的是加速度和回零上升，不是 T0 跳到 50。

回零边（命令 ≈ 0）在原 02 图上堆成一列 5–50 ms。那些边进了 `edges_from_svg.csv`，没有进上表。

### 加速和上升

原图 01 / 03，汇总 [S06](S06_accel_and_rise.png)。

- 线性加速帽画在 **2.0 m/s²**。40 mm/s 标签是「峰 |a| 还靠近这条线」，不是延迟突变。
- 过了 2 m/s²，延迟仍在 30–45 ms 附近；80 mm/s 到约 4.3 m/s²。
- t10–t90 多数 < 75 ms。低加速有一条约 185 ms；高加速有一条约 130 ms（回零）。

### 稳态 Kp

[S07](S07_kp_vs_speed.png)。202437 的 04 图是单色，正负号分不开；每个速度两个点。

- ≤12 mm/s：大约 0.82–0.94，SNR，不是第二套植物。
- ≥20 mm/s：大约 0.96–1.01，高于旧标注 0.892。

## 抖动和 Td 区间

| campaign | n | age mean / p95 / max | dt mean / p95 / max | Td 当区间？ |
|---|---:|---|---|---|
| 202437 规范 | 42414 | 2.69 / **6.29** / 7.54 ms | 5.14 / 5.88 / 12.3 ms | 是（p95 > 1 拍） |
| 200549 较早 | 45567 | 2.53 / 5.78 / **31.9** ms | 5.01 / 5.01 / **35.2** ms | 是；max 差很多 |
| synthetic | 44700 | 3.08 / 3.10 / 3.10 ms | 5.00 / 5.00 / 5.00 ms | 否（假数据） |

202437 上 `feedback_age` 和 `sensor_age` 统计相同。走廊 / 证书应留年龄裕度。TDPA 记账仍用同一拍 `F × v_cmd`，不被这点抖动拆掉。

## 接触（夹具，开环 `--tdpa-press`）

力环关着。慢寻然后恒速压。不是 hybrid `F*` 跟踪。

| 压 | v | 窗 | ΔF | Δx | Ke 窗 | Ke 落定 | 符号 | E_obs 上升比例 |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| 210139 | 8 mm/s | 0.185 s · 38 tick | 2.10 N | 1.48 mm | 1420 | 1420 | ok | 1.00 |
| 212852 | 3 mm/s | 0.535 s · 105 tick | 4.05 N | 1.61 mm | 2522 | 2199 | ok | 0.98 |

两条都 `alpha_p95 = 0`。原 F–x 见 [`original_campaign_plots/08_contact_press.png`](original_campaign_plots/08_contact_press.png)。[S04](S04_contact_ke.png) 只连了 `identification.json` 里的端点——中间轨迹回不来。

空气 `tdpa_shadow.json` **不是** 刚性压证明。202437：Fz 均值 0.25 N，`E_obs` 在 −3.0…+11.3 mJ，`alpha_clamped_ticks = 15120`。那是空气残差被当成负载。

`tdpa.apply: true` 加上现在的 `e_adm = F* − Fc` 和 `alpha_max = 400` 会在 D=25 时把导纳写反。保持 `apply: false`。

## yaml 现在怎么用这些数（2026-09-02）

对照 [`yaml_vs_id.csv`](yaml_vs_id.csv)。

已经对上的：

- `safety_shield.plant.t0_s = 0.035`，`tp_s = 0.012`
- shield `mode: observe`，`stop_dx` 未 certified
- `cdyob.mode: "off"`（必须带引号；YAML 1.1 里裸 `off` 是 false）
- `tdpa.enabled: true`，`apply: false`

不要当成测到的 T0：

- `system_delay_s: 0.055` 和 `force_barrier.t_react_s: 0.055` — 占位，对齐 t_react
- `cdyob.t0_s: 0.030` — 过时；CDYOB 关着，也别往这里写 T0

辨识时的包络建议 vs 现在的 yaml：

| 建议（`id_reference.log`） | 现在的 yaml |
|---|---|
| 空气 / 首触 10 mm/s | `press_envelope.first_touch_m_s = 0`（不用）；R2 空气走 `v_seek_free_m_s = 0.020` |
| 确认接触追 20 mm/s | 包络 0；确认接触落到 `max_vz = 0.080` |
| 超力回退 25 mm/s | 回退 `u_retract = 0.080` |
| 再接触压 ≈ 2.8 mm/s，夹到 8–12 | `recontact_vz_cap = 0.008`，保持 0.22 s |

`ke_initial = 80`，`ke_cap_ub = 2000`，`k_ub = 8000`。2000 的帽低于慢压夹具的 2199–2522。`k_ub` 只用来估再接触速度，不是组织 Ke。

## 明确没辨识的

1. **力环 F→v FRF / `Ya`。** 测的是速度跟踪 `Gv`。hybrid `F*` 跟踪是错的激励。
2. **单组织 Ke。** 上面两个数是垫。
3. **停-反 `Δx_b^ub`。** shield 保持 observe。
4. **202437 单段 chirp gain 和单段 Tp。** 原始 `chirp_fits.csv` 已删。
5. **带 `t_edge_s` 的完整 `edges.csv`。** SVG 还原没有时间戳，04 图没有符号。

## 分析器入口（不要重跑，除非要求）

- 空中：`python -m peirastic.apps.identify_plant --air-campaign …`
- 打包：`python -m peirastic.apps.identify_plant --identify --air-out <air_dir> --analyze-tdpa <press.csv>,…`
- 图清单：`identify_air.REQUIRED_PLOTS`（`01`–`07`）。`08` 是接触打包时画的。
