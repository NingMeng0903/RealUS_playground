# 50 mm 探头：实际执行联合 QP

2026-09-10 更新：本页保留 V7 的原始运行说明。用户新授权的 4–4.5 N 修复策略、主动倾斜分配和有限尝试采用独立 `active_probe50_v8.yaml`，详见 [V8r2 公式、验收与运行命令](V8_ALLOCATION_AND_TEST.md)。

配置：`peirastic/config/contact_qp/active_probe50.yaml`。TCP 位于接触面中心；半孔径 25 mm；图像左侧是 TCP +X，深度是 +Z。图像配置与现有 baseline/shadow 特征进程一致。

本次切换实际启用外环法向修正、绕 Y 的 rocking 和 α 进度调节。原工具 Z 轴目标力仍为 4 N，保留原扫描准备、寻触、监督停止和录制流程。shadow 的建议没有执行过，实际声窗改善与力误差仍以这次新记录判断。

## 启动顺序

1. V7 修正了 active 把执行预测速度写回外环命令状态的问题，并使外环指令加速度使用与名义力控一致的实际周期。需要在原 controller 终端停止旧进程后，启动本版本。不要同时运行两套 controller。本操作由用户执行；开发过程没有自动启动或重启设备。

```bash
cd /media/camp/EXT_DRIVE/ICRA_YM/script
bash run.sh controller
```

2. 当前 confidence 进程可以继续运行，无需更换配置或重启超声发布者。已核对的窗口版本是 `8b51133835031ba9ee2d`，配准版本是 `fbe3a5c9d4cb3065966a`。若超声 publisher 或 crop/hflip 改变，需要重新核对配准版本。

3. 按原流程确认手柄和数据服务就绪，在扫描终端执行：

```bash
cd /media/camp/EXT_DRIVE/ICRA_YM/script
bash run.sh check
bash run.sh record --force-profile icra --speed-m-s 0.005 --keep-raw \
  --contact-qp-config /media/camp/EXT_DRIVE/RealUS_playground/peirastic/config/contact_qp/active_probe50.yaml \
  --data-root '/media/camp/EXT_DRIVE/ICRA_2027/icra 2027_contact/real_characterization/active_probe50'
```

按原提示示教和 Enter。首轮先完成一条直线，下一条提示时输入 `q`。这次 QP 建议会实际参与控制；与先前 shadow 不同。若仍出现 QP 拒发、力反馈过期或地轨通信异常，保留该 attempt 和 controller 输出，结束本轮；不要靠反复 Enter 或放宽门限尝试通过。

V7 的软件修正不等于真实扫描已验收。active/001 的第 003 次最终复核拒绝尚缺旧版精确原因，时序疑似超过 10 ms；本版保留该拒发保护并补全原因与时刻。地轨 FA24 / Modbus 异常也不因外环状态修正就视为消失。详见 [本次故障分析](ACTIVE_FAILURE_INTERFACE_V7.md) 与 [原始记录摘录](active001_failure_evidence_v7.json)。

## 配置与测量时间

标称测量周期保持 5 ms，没有为绕过检查把它改成 6 ms。active 的新测量滤波按真实源时间间隔推进，重复样本不重复更新；源间隔与测量年龄分别有 15 ms 上界，超界、倒序或源身份变化仍拒绝。旧固定时基模式保留。控制器必须宣告 `contact_qp.source_timebase_bilinear_v1`，旧进程在扫描准备前被拒绝。

这是滤波时基兼容修复，不是放宽原机器人保护。固定 5 ms 下与原 Butterworth 滤波的等价检查，不代表非均匀采样时真实力误差已经获得零退化证明。

首轮不启用未认证的机械能量约束：`energy_constraint_enabled: false`，`physical_w_checked: false`。原机械保护继续运行；没有把软件模型作为物理无源认证。

## 记录后检查

保留整个新 session 目录。检查最终 QP 命令、发布事实、实测运动和配准图像响应；确认修复是否实际执行且改善对应声窗。按扫描单元对比 4 N 力误差、实际缺口长度与覆盖，并保留失败记录。此前 baseline/shadow 的起终点重新示教过，不能自动当严格同路径配对试验。

关键时基审核：[review_irregular_source_v5.md](review_irregular_source_v5.md)。

## V7 本次验证

medium 编码、ultra 定向接口审核 PASS。最后修改后，外环全套短回归 584 项通过，唯一 native 启动测试在当前沙箱内失败、沙箱外相同代码复查通过；原始失败记录保留。原力控八组回归 46 项通过。active 配置离线校验通过，没有连接硬件。详情与命令见 [V7 故障与验证报告](ACTIVE_FAILURE_INTERFACE_V7.md)、[ultra 审核](review_active_interface_v7.md)。

这次没有改变 4 N 目标或增加 1 N 的修复裕量。实际扫描完成性、声窗改善、力精度和 003 的时序长尾仍待实测验收。

V7 源码快照为 `active_probe50_source_v7.zip`，身份与运行环境见 `active_probe50_source_manifest_v7.json`；下面的 V6 记录与旧压缩包作为历史保留。

## V6 历史验证

代码使用 GPT-6 medium，关键时基／状态／旧服务拒绝采用 ultra 定向审核，限定软件范围 PASS。

- 最后代码修改后 `peirastic/tests -k 'not 100000'`：579 passed，3 个此前已跑长测试 deselected，20.76 s。
- 原 force/contact/observer/Ke/guard/TDPA 八组回归：46 passed，11.62 s。
- medium 针对时基／接线 40 passed；原扫描入口 33 passed。ultra 独立定向 41 + 3 passed。这些检查有重叠，不累加成独立测试总数。
- 真实 shadow 的 13,738 条控制记录时间戳重放通过，105 条重复测量没有重复更新；独立滤波 oracle 对照通过。
- 真实 active 配置 `--validate-only` 通过，未连接硬件。源码范围 whitespace 检查通过；用户采集的数据日志未清理或改写。

JUnit：[外环回归](peirastic_regression_active_probe50.xml)、[原力控回归](force_regression_active_probe50.xml)。源码增量及配置保存在 `active_probe50_source.zip`，文件身份见 `active_probe50_source_manifest.json`。原 V4/V5 历史交付保留。
