# Runtime 最小适配图 v2：原内环不动，先真实特性 shadow 采集

**状态纠正：已撤回初版“最终 17359 元数据完全兼容”的结论。** root 的真实 H5 冒烟发现 publisher 最终 shared metadata 合并覆盖 clock_domain，旧 worker 会拒绝；失败记录保留于 [`real_frame_interface_pre_fix.json`](real_frame_interface_pre_fix.json)。下文第“17359”节已修正；真实回归通过前不宣称该输入接入可用。重复源控制推进的详细接缝以 [`MEASUREMENT_ADVANCE_MAP_V1.md`](MEASUREMENT_ADVANCE_MAP_V1.md) 为补充，不能跳过整个 force compute。

本文件是只读代码核对与接口设计，不是已实现功能。优先支持用户主动启动的真实扫描记录：原控制器产生并执行原命令，新 QP 仅计算候选并记录。停止依靠合成声学结果反复调参。active 接入是同一模块的另一个显式模式，必须有装置几何/坐标标定；禁止在真机工厂调用 `ProbeGeometry.synthetic()` 兜底。本文不授权启动、停止或改变现有 `run_controller` 进程。

范围严格限于 payload/外环工厂、runner 适配和日志。原 `JointIkController.update/commit_publication`、native 协议/求解器、rail worker 算法保持原样；此前 generic Cartesian native 约束接法不再是本设计前提。实际残差用于监测，不能把微小 IK 残差当作严格 H 进度认证失败。

## 现有入口与最小工厂

| 文件 / 函数（核对时行号） | 现有行为 | 最小适配 |
|---|---|---|
| `peirastic/api/payloads.py:HfpcPayload.to_json:174–189` | law 只允许 tff/admittance/fce；非 admittance 开启 TFF | 加 `contact_qp`；用一个显式执行选择 `shadow/active`，默认 shadow。保留原 force/path 参数。 |
| `peirastic/realman8dof/session.py:compile_request:379–401` | icra_path 构造真实 `ForearmReference(payload['path_spec'])` | 保留该参考与原工厂调用；不给 contact_qp 改用 LinearScanReference。初版只支持 icra_path，其他 source 明确拒绝而非悄悄换轨迹。 |
| `peirastic/realman8dof/modes/track.py:build_track_hybrid_phase:354–404` | 先建原 force_law，再编译 Cartesian position，再包装 HybridTffOuter | shadow 保留完整 HybridTffOuter，附加无输出权限的候选观察器；active 在这里选 ContactQpOuter，复用原 position/force_law。 |
| `track.py:_hybrid_force_law:238–258` | 原 LegacyForceLaw + TorqueTilt | 保留其 nominal，不能另外实例化另一套增益/动态。`force/config.py:_SKIP_PAYLOAD` 已跳过 law。 |
| `session.py:ProxyOuter:493–575` | `__getattr__` 透传到当前 child；sample 每次读当前 child | 可以透传回调，但必须在本拍开始捕获具体 owner；回调时不能重新查询 proxy。 |
| `daemon.py:_install_velocity:998–1030` | on_step 内 bind 新 child，set_origin(t_s=当前 runner t_ref) | active outer 保存此绝对时钟偏置；本拍旧 owner 的 publication 必须在 on_step 之前完成。 |

shadow 不能通过调用 `nominal.prepare()` 或共享 force controller 再算一次来获得候选；这会重复消费样本或改变原动态。最小旁路点是 runner 原 `outer.sample(...)` 返回以后（loop.py:6768–6771），把本拍**已经生成的 nominal twist**、原 path/反馈、源 wrench/pose/图像快照送给独立候选计算器。最终传给 `inner.update` 的仍是原 twist。计算/图像提取宜放在有界后台队列；候选迟到只记录其对应旧 sample/proposal，不能回写本拍命令、原 governor 或参考时钟。

## active 外环必要接口（新增设计，不是已有 API）

`ContactQpOuter.publication_owner` 返回 `self`，`owns_reference_clock=True`。runner 通过 proxy 只查询一次 owner，然后固定其 `prepare/on_publication/abort` 绑定方法和元数据。`prepare(...)` 使用原 sample 的测量参数，额外接收独立的 `wrench_sample_id/source_time/fresh`；返回待给原 inner 的 6D twist，并保存本拍 proposal ID、H、alpha、源时刻、工具姿态与 nominal command transaction。`on_publication(success, final_prediction, dt_actual, facts, reason)` 只处理自己的 pending proposal；`abort(reason)` 必须幂等，仅丢弃命令 pending，不重置/回滚力滤波、接触判据、已消费 ID 或实际运动历史。

现有底层依据：`force/legacy.py:34–63` 的 prepare/commit_applied/abort；`force/torque_tilt.py:205–247,475–505`；`force/nominal_transaction.py:97–129,226–234`。底层 abort 在无 pending 时会报错，所以外环必须守住 exactly-once，不能把每个 fault 分支都无条件调用底层 abort。已提交成功后，也不能因为后续 logger/on_step 出错而“撤销”刚才设备接受的动作。

## runner 的原发送链和所有失败出口

所有以下 runner 锚点位于 `rm75_control/rm75_control/control/joint_admittance_8dof/loop.py`。主函数 `run_joint_admittance_phases:6225` 已调用 `inner.update(..., commit_history=False, auto_commit=False)`（6853–6893）；无需更改 IK 更新/提交算法。

| 位置 / 分支 | 本拍设备事实 | 新 outer 的处理 |
|---|---|---|
| tick 顶部 stop/watchdog、duration/max_duration，6470–6509 | 尚无新 proposal | pending 为空时不消费/回滚测量；离开时丢弃意外残留 pending。 |
| rail feedback fault/coast、sensor stale/coast，6559–6689 | 尚未准备本拍命令，旧设备动作可能继续 | 不假定速度为零；记录缺测/旧命令状态。无新 proposal 不调用 nominal prepare。 |
| prepare/observer 抛异常 | 尚未发送；测量可能已被消费 | 新 mode 调现有 `_fault_stop`，仅 abort pending；不再重放同源样本。 |
| `_guard_uncertified_brake_before_inner:6810` | 新 arm/rail 均未发送 | 现有 stop 保留，通知 captured owner failure。 |
| inner.update 抛异常、`_guard_qpik_step_before_send:6922` 拒绝 | 新 arm/rail 均未发送 | stop 后调用原 `inner.abort_publication`，通知 outer failure。 |
| `native_timeout_coast`，6928；client.py:605–622 | native 仍有旧 inflight，旧动作可能继续 | **仅 active contact_qp** 使用原 stop/abort 并结束 mode，不能继续把晚回旧解提交给新 proposal。其他 mode 原 coast 行为不改。此最小选择已获 root 确认。 |
| stop/watchdog/post-solve stale，7047–7089 | arm 未发送，rail 未 reserve | outer failure；原停止/inner abort 保留。 |
| rail reserve 拒绝/异常，7128–7144 | arm 未发送；rail 请求拒绝或结果未知 | 保留 `_reserve_rail_target` 的 stop；带 reason 通知 outer，不能宣称物理 rail 已停止。 |
| arm send 异常，7160–7175 | **arm 是否到达驱动未知**；rail reservation 执行原 abort | 标 `arm=unknown`，不写 `not_sent`；outer abort、clock 不前进。 |
| arm 成功、rail commit 返回 false，7180–7189 | `arm=API_accepted`，`rail=commit_rejected` | 保留 `PARTIAL_ARM`，outer abort/clock 不前进；不能回滚已发 arm。 |
| rail commit 或 inner commit 抛异常 | 根据此前返回值保留 accepted/unknown 事实 | 新模式 catch/stop/清理 pending；不把“软件事务失败”改写为“设备没动”。 |
| 两个 transport 接受，inner commit 完成，7191–7194 | 软件发布成功，物理执行未确认 | 在此立即回调 captured owner；随后才进入 governor/clock 和 daemon on_step。 |
| 外层 KeyboardInterrupt/finally，7349–7360 | 可能部分发布 | exactly-once 通知 unresolved proposal，保留事实；不重置 observer。 |

`_reserve_rail_target:5976–6021` 对不支持 reservation 的旧 bridge 会 fallback 到立即 `set_target_m`。active 工厂/runner 必须在配置期确认 enabled bridge 有 `reserve_target_m/commit_reservation/abort_reservation`，否则拒绝该 mode；这不需要更改 rail worker。rail `commit_reservation=True` 表示 worker 可见的命令接受，不是驱动执行 ACK。CANFD proxy 的 `write` 也只是代理接受；日志必须记录 SDK/proxy 来源。

`final_prediction` 应同时保存 `J(q_meas) @ step.qdot` 的 TCP/base 指令预测、现有 `step.v_tcp_estimated` 的执行模型预测，以及最终 `rail_pub_m/qdot0_pub`、target 是否被 `_wall_clock_rail_target` 修订。两种预测不能冒充实测。若 target 被修订或 rail coast，记录 prediction 适用性，不用 Jqdot 宣称精确完整执行。实测取已有 full measured tracker 的独立通道。

## 约 70 行 runner 修改设计

以下是锚点级伪 diff；省略部分为**原代码原调用**，不是新实现替代。`PublicationFacts/WrenchSourceCache` 是拟放在新 adapter 模块的薄数据辅助，不包含 IK、native、能量认证或 rail 算法。失败回调内部调用 outer.abort；exactly-once pending 守卫集中处理，不散落重置 controller。

```python
# runner scope, beside the existing watchdog/_fault_stop setup
pending_owner = None
pub_facts = None
wrench_cache = WrenchSourceCache()  # records existing updates in other modes too

def notify_outer(success, reason, final_prediction=None):
    nonlocal pending_owner
    owner, pending_owner = pending_owner, None  # capture before any callback/swap
    if owner is not None:
        owner.on_publication(success=success, reason=reason,
                             final_prediction=final_prediction,
                             dt_actual=dt_wall_actual, facts=pub_facts.snapshot())

# append to existing _fault_stop AFTER original arm/rail stop requests
notify_outer(False, reason)

# tick: capture before obs.update; shadow exposes no publication_owner
owner = getattr(phase.outer, "publication_owner", None)
tick_outer = owner if owner is not None else phase.outer
if owner is not None:
    t_ref = owner.reference_time_s  # includes hot-install absolute offset
source = (id(async_obs), getattr(state_bus, "session_id", None),
          float(snap.t_s), int(getattr(snap, "wall_time_ns", 0)))
fresh = wrench_cache.is_new(obs, source)  # t_s, NOT relay seq
if owner is None or fresh:
    ...  # original obs.update -> raw/filtered link7 -> current TCP conversion
    wrench_cache.store(obs, source, f_ext, f_ext_raw, relay_seq=snap.seq)
else:
    f_ext, f_ext_raw = wrench_cache.copy_latest(obs, source)
sample_fn = tick_outer.prepare if owner is not None else tick_outer.sample
...  # original kwargs selection, inspect this captured bound method
if owner is not None:
    sample_kwargs.update(wrench_sample_id=wrench_cache.source_id,
                         wrench_source_time_s=snap.t_s, wrench_fresh=fresh)
    pending_owner = owner
    pub_facts = PublicationFacts(owner.proposal_generation)
twist = np.asarray(sample_fn(t_ref, pose_pin, f_ext, **sample_kwargs))
if owner is None or fresh:
    ...  # original update_leftover block; duplicate source does not advance it
...  # optional shadow queue receives snapshots; never replaces twist
...  # original brake guard, original inner.update(auto_commit=False), QPIK guard
if owner is not None and path_reference_should_freeze(step):
    _fault_stop("contact_qp_native_timeout_coast")
    inner.abort_publication()
    phase_stopped = True
    break
...  # original post-solve freshness/stop guard and rail_pub_m calculation
if pub_facts is not None:
    pub_facts.note_rail_request(rail_pub_m, qdot0_pub, rail_coast_active)
...  # original reserve call; failure already invokes _fault_stop
if pub_facts is not None:
    pub_facts.note_rail_reservation(rail_ok)
...  # before original arm call, mark arm=attempted_unknown
...  # original arm call; on success mark arm=API_accepted, exception stays unknown
...  # original rail commit; record returned false/true or unknown on exception
...  # original inner.commit_publication(step.qdot), unchanged
if owner is not None:
    prediction = owner.prediction_snapshot(inner.kin, q_meas, pose_pin, step,
                                           rail_pub_m, qdot0_pub, rail_coast_active)
    notify_outer(True, "accepted", prediction)
    t_ref = owner.reference_time_s
    scale = raw_scale = owner.last_alpha_committed  # telemetry only
else:
    ...  # original _reference_governor_scale/filter/ramp/freeze/t_ref increment
...  # original logger, phase.on_tick and on_step (daemon may swap here)

# new-mode exception handling in existing outer try/finally boundary
# if unresolved owner: issue existing coordinated stop, inner.abort_publication,
# notify failure once; retain all observed states and partial publication facts.
# finally also clears any unresolved owner on external exit/KeyboardInterrupt.
```

新增 code review 要求：`pub_facts` 每拍重置，避免 legacy tick 继承上拍事实；callback 的成功/失败路径都不能抛出后又被重复通知。实际实现需对所有 existing break/exception 落点作测试，不能只覆盖成功路径。 `_fault_stop` 先停设备再通知/记录，避免 native abort 等待旧 reply 或日志 I/O 延迟停止。

## 参考时钟与实际残差

active 的参考时钟只由 publication callback 更新，局部时间限定 `[0, reference.duration_s]`，runner 对外用 `origin + local_time`。`set_origin(t_s=...)` 保存安装偏置，不把 daemon 的全局 t_ref 突然重置为 0。参考时钟不是实际采集弧长；实际覆盖必须用实测 pose/V 与图像时刻单独登记。

`loop.py:7200–7220` 是旧 governor/soft-start/freeze/time increment 的唯一分支，**只在 active owner 存在时跳过**。旧 inner 的关节/滑轨/碰撞/加速度保护不跳过。shadow 和其他模式保留旧 governor 与原路径前馈。active 不复用严格 `accepted_alpha` 作为真机 H 认证；名义 H 与 Jqdot 的小残差是执行误差监测量，不能据此以 `off-H` 立即停止或把全部进度归零。可记录正向投影与离开子空间的残差，使用有界、非负、≤本拍请求的参考推进策略；其物理精确性不在本次声称范围内。

真实 ForearmReference 启动时 b=0，现 `geometry.accepted_alpha` 在 zero_path 返回 0；若机械式照搬就会使起步参考永远停在 0。outer 必须区分“参考时间参数推进以启动 ramp”和“实测空间进度”，终点与暂停也分别处理。小数尾差/有限区间收尾应有显式容差，不重造 `LinearScanReference`。

`ContactGatedReference.sample:190–226` 会自己积累输入时间。active 不可再加另一个独立墙钟；可复用接触/持续 4 N 判据，但 gate.elapsed 必须来自已接受的参考时钟。daemon `_on_step:1160` 用 `contact_gate.started/elapsed_s` 判断有限扫描结束；adapter 必须保留该可见接口或显式新增 owned-clock 分支。shadow 保留原 gate 不变。

## 源测量去重：t_s 是源接收时间，relay seq 是重发布序号

`AsyncStateSnapshot`（`admittance_common/async_state.py:51–60`）有 force_raw、t_s、wall_time_ns、seq。直连 `_on_state:263–291` 同一次 UDP 回调产生 wrench/关节和主机 monotonic/wall 时间，`_store_snap:171–180` 增加源 callback seq。没有独立硬件 force acquisition timestamp/force_seq，不能声称已有。

relay 情况不同：`state_relay.py:_publish_snap:722–767` 每次读取缓存都增加 publisher seq，`_write_slot:191–203` 把这个 pubseq 写入 snapshot，却保留旧 `snap.t_s/wall_time_ns`；`_publish_once:770–777` 甚至可能 watchdog_hold 重发旧 snapshot。因此 **seq 增加不表示新 wrench**。`RelayStateBus.session_id:817` 可区分重连代际；当前 `full_measurement.measurement_id=snap.seq` 不能直接用作 nominal wrench ID。

新 active 模式以 `(observer identity, relay session/source identity, source t_s)` 判断样本，要求有限、fresh、严格递增的 t_s；wall_time_ns 与 relay seq 用于审计，不用可跳变 wall clock 控制状态积分。同源同 t_s 即使 seq 增加也复用缓存。t_s 回退/同身份内容变更按无效样本记录，不补零、不重新消费。源时钟改变时清空插值关联并结束/重新安装该外环代际。

`CompensatedForceObserver.update:219–263` 每次 append pose/time、增加 n_updates、推进 regressor/LPF，完全没有重复源保护。runner 当前无条件调用，所以仅给 nominal.prepare 去重不够。最小解是在**新 active 模式**的 runner wrapper 前缓存 raw/filtered compensated TCP 输出，重复源不调用 `obs.update`，也不调用 `update_leftover`。缓存也旁观 legacy 成功 update，用于 hot-install 时避免同一个共享 observer 再消费最后一帧；不改变 legacy 调用次数/输出。首次无缓存且没有新源时等待新样本，不能给名义交易编造递增 ID。

shadow 要保证原控制器动态不变，因此**不启用原执行链的去重改动**。shadow 的独立分析支路按上述源时刻去重；记录原控制器实际用了几次该样本。若需要消除旧 observer 重复消费来改善执行，这是另一个显式变更与对照，不能偷偷混入“原控制器 shadow”。控制力始终使用最新源 wrench，不减 ultrasound 的图像延迟。duplicate source 如何保持已有命令/等待新的 nominal prepare 由 outer 明确记录；不能反复 prepare 同 ID。

## 真人日志的最小追加边界

现 `StepLogger` 已有 6D `fx..tz`、6D raw-comp、6D requested/achieved、pose、q_send/q_meas、模型速度、wall dt、arm/rail 发送时刻。位置：loop.py header 3750–3783/3900，`write:4360` 是有界队列，`_snapshot_outer:4257–4347` 只复制白名单，`_write_impl:4450/4830` 在 worker 格式化。不要再给控制线程写 H5 或做图像提取。新增 `contact_qp_record` 为固定长度向量+标量快照；必须同时扩展 snapshot 白名单与 worker 输出，否则给 outer 加属性也不会落盘。

| 最少新增记录 | 精确来源 / 用途 |
|---|---|
| run_id / owner generation / proposal_id / mode=baseline,shadow,active | 一个候选绑定一个 nominal 和原源帧，热切换不串 owner |
| source_monotonic_s、source_wall_ns、relay_session、relay_seq、wrench_source_id/fresh、control_tick_mono_ns | snap.t_s/wall_time_ns、state_bus.session_id、snap.seq；区分采样与控制/发布 |
| W_control_tcp[6]、W_comp_unfiltered_tcp[6]、W_physical_tcp[6]+status | 前两个来自现 f_ext/f_ext_raw；物理语义另见下节，不能自动全负 |
| V_nominal_tool[6]、V_candidate_tool[6]、V_inner_input_tool[6] | 原 sample 结果、shadow QP 输出、实际 inner 输入；shadow 后两者应保持 candidate 与执行分开 |
| V_command_prediction_base[6]、V_model_base[6]、V_measured_base[6]+valid/fresh/metadata | Jqdot、step.v_tcp_estimated、full measured tracker；不能把旧 twist_achieved 名称当完整实测证明 |
| q_send_arm[7]、qdot_inner[8]、rail_target_published_m、rail_v_ff、rail_target_clipped、arm/rail publication states | 原发送局部变量，不以请求或 reserve 代替最终发布事实；失败也必须落盘 |
| alpha_candidate、alpha_committed、reference_local_s、actual_forward_coordinate、residual[6] | 参考与实际运动分开；shadow alpha 不控制执行进度 |
| image publisher instance/frame_seq、capture/receive/effective mono时间、delay+uncertainty、clock/version/crop/flip/window/algorithm/calibration revisions、qL/qC/qR及valid | 已有 ContactObservation/FeatureConfig 和图像 metadata；image_effective_time 只对图像计算一次 |
| geometry T_tcp_face、half_length、frame/sign/force_model/kinematic hashes、port_status、dropped_records | run manifest 存一次，逐拍引用revision；缺值显式 unavailable |

现真人记录器 `/media/camp/EXT_DRIVE/ICRA_YM/script/record.py:193–226` 已将 raw sensor wrench、compensated wrench_tcp、source times、session 与 seq 写入 H5，并以 `(session, source timestamp)` 去重；`receive_image:235–261` 保留 JPEG、publisher instance、frame_index、原 metadata_json、capture_monotonic_ns、source_time_ns/shared clock。优先将新 controller sidecar 通过同一 clock/run_id 离线关联原 H5；不为了新增候选记录修改 image publisher 或机器人 state/native 协议。保存原图即可离线重算特征；如果实时候选尚无特征，记录 `image_unavailable`，不能以模拟图替代真人图。

已有主日志 `P_ext_trans` 只累加前三个分量，且 f_ext 在 TCP/tool、旧 achieved 在 base，不能拿它直接当 6D 同端口机械功证据。新日志必须先明确同一 TCP 参考点，将 W/V 一起转到同一坐标再算 `W·V`，保留原值与有效状态。

## 物理 W 的现有语义和不可猜测部分

`force/compensation/v2/frames.py:17–25,123–153` 的 FrameContract 明确声明 `environment_on_tool`。`configs/force_sensor.yaml` 当前六轴 raw sign 均为 −1，注释说明这是 sensor→物理语义的原标定转换，不能在它之后再对 f_ext 整体取负。

`observer.update` 中 `_signed = wrench_sensor_to_link7(force_raw, contract)` 仅完成 raw sign/旋转/力矩平移，**尚未扣掉 payload 重力/惯性/bias，不是外部接触 W**。随后 `residual = signed - W_model @ phi` 才是模型补偿的 link_7 环境 wrench；`f_ext_raw_last` 是未滤波 residual，返回的 f_ext 是其因果 LPF。runner 再通过 `kin.wrench_link7_to_tcp`（model.py:327–340）减去 `r_LT × f` 并转到工具坐标。此路径中没有 controller axis_scale，也没有统一“压缩正”翻转。控制器随后在 controller.py:1391–1403/1526 根据期望力符号生成一维 `normal_sign*f_ext_z`；这是控制标量，不能替代 6D 物理 W。

因此可记录 **依据当前 FrameContract 与 payload model 得到的 W_physical_candidate_tcp = 已补偿、转TCP的 residual**，以及单独的 filtered control wrench/normal scalar。只有当前实际探头/TCP配置与该标定契约一致、方向/参考点已核对且源时间有效，才能标为可用于端口监测的 W。当前代码中的 semantics 字符串和历史 verified 注释不自动证明本次探头装配/声学面标定已经完成。缺失或冲突时 `physical_port_status=monitor_unavailable`，物理 W/功字段用 NaN/null 并保存原因，仍保留 raw/compensated candidates；**不全负猜、不补0、不宣称无源认证**。过滤后的 W 与当前V也不能忽略相位/时刻差来算真实瞬时功。

## 用户可主动取得的实物证据与验收重点

需要用户实测/确认并记录的是：当前 TCP→声学面中心刚体变换与孔径尺寸；图像左右与物理 rocking 的符号及裁剪翻转；当前 force sign/传感器→link7→TCP/payload 绑定；ultrasound 有效时延与波动；实际 arm/rail 速度和命令延迟/误差。shadow 可以先采集原始证据，不要求先证明声学收益，也不以未标定 active 催促上机。用户自行启动原扫描/已有小幅特性采集入口；代理不启动硬件，不改变现有运行进程。

最小测试复用位置：`peirastic/tests/test_contact_qp_proxy.py`（透传及owner固定）、`test_contact_qp_nominal.py`（abort保留测量、重复源）、`test_daemon_boundary.py`（hot install/contact elapsed）；`rm75_control/tests/test_final_qpik_send_chain.py` 的发送/guard mock；`test_rail_reservation.py:_ready_bridge` 的 reserve/hold/commit失败 mock。新增适配测试重点是 shadow inner输入逐拍等于原输入、旧mode governor不变、relay换seq但同t_s不重复分析消费、部分发布事实不丢、active timeoutcoast停止、callback恰一次、真实记录字段/时间关联和缺失物理W的unavailable，而不是给现有IK算法新增功能。

## 17359 真实图像元数据兼容核对

**2026-09-10 真实 H5 回放纠正**：初次核对漏掉了 publisher 最终的 shared metadata 合并，原“已经兼容”结论撤回。`peirastic/apps/contact_qp_features.py:20–38` 原要求 clock_domain=host_monotonic，但 `UsImagePublisher.send:191` 的 `meta.update(self.clock.metadata(...))` 最终将顶层 clock_domain 覆盖为 `realus_shared`（`realus_clock.py:131–134`）。原 capture_monotonic_ns 仍为 host monotonic，当前 worker 严格按顶层字段判断会把真实帧全部拒绝。最小修复由 high 在 feature 解析端完成：识别已核对的 realus.us_framegrab/schema1/host_frame_read_complete 契约，校验明确 capture 字段与 shared header，不能把 epoch source_time_ns 冒充 capture。无需修改原机器人或图像发布协议。

| 必需项 | 上游证据 |
|---|---|
| 地址/主题 | `us_framegrab/configs/config.yaml:23–24`：17359 / amongus_camera_frame_v1 |
| source_id | config.yaml:29 = realus.us_framegrab；`src/us_framegrab/zmq_pub.py:77` 写入 |
| publisher_instance_id | zmq_pub.py:114 创建 UUID；93/187 写入帧；不依赖静态 session_id |
| capture_monotonic_ns / capture_wall_time_ns | `runtime.py:228–230` 在成功 read 后、crop/JPEG 前取时；264–265 传给 publisher |
| clock_domain | 初始 zmq_pub.py:186 为 host_monotonic；**最终** 191 的 merge 覆盖为 realus_shared |
| frame_index/crop_box/hflip | zmq_pub.py:80/94–95 与 runtime.py:262–268；当前配置 hflip=false |
| 测试现成入口 | `us_framegrab/tests/test_zmq_pub.py:58–109` 验证这些元数据和实际 send 的传递 |

真实复现：`uncalibrated/chenwei/LH_Per_C_DtP.h5` 第 0 帧/源 frame_index=13931，保留原 metadata+JPEG 调用旧 process_parts，得到 `ValueError: monotonic capture timestamp missing; cannot align control history`。其 capture_monotonic_ns=1048465350605926，顶层 clock_domain=realus_shared。完整字段与 SHA-256 见 [`REAL_CAPTURE_ENTRYPOINTS.md`](REAL_CAPTURE_ENTRYPOINTS.md)。没有订阅端口或检查/重启正在运行的进程。修复后应以原 JPEG 与最终 metadata 的回放验收，用户启动时报告有效帧/错误计数；不能只无限 warning+丢帧。

时延语义：publisher 的 `capture_monotonic_ns` 是**未经 time_offset 修正的 raw 主机 read-complete 时间**；`time_offset_ns` 只加到 source_time_ns/shared header timestamp（zmq_pub.py:171–196）。当前配置 offset=0。现 live worker 基于 raw capture 减配置的有效延迟一次是正确的数据路径；即使将来 header offset 非零，也不能把它又加到 raw capture 后再无条件减一次。worker 应记录 raw/offset/使用的 delay/有效时刻及版本，明确选择一种基准。

目前 `process_parts` 是 raw-live 输入接口，没有 already-aligned H5 接口。离线已对齐输入应走 `FeatureExtractor.extract(..., already_aligned=True)` 或显式的等价注册路径；不得把 aligned timestamp 伪装 raw capture 再减 152 ms。已有 `FeatureExtractor.effective_image_time` 支持不重复减时延，但调用入口仍须明确标记。测试应覆盖 raw、header offset 非零但 raw capture 不变、以及 already-aligned 三种情况，不根据文件名猜时间基准。

本次核对源指纹：contact_qp_features.py `1f498e2e8524c200ac8cf1f17d5c7a9d33f0ba1c2593685e33975d1fcb326c9e`；us_framegrab/zmq_pub.py `00dcf34682f5ec793ae0c2cc176b68cd1cd6056f141b9bddc1f1f286ca2ca36d`；us_framegrab/runtime.py `e4ee3abaf7d00273c33847a194e0ab46656df53aea5e8e808a398e14a0c70963`；force_sensor.yaml `5b6a55b27dfe6722f4299c7c90d9a80cbfb6bdff23cd408fff47dc0765ae9c2c`。
