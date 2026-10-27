#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ablation_harness.py — FCCS 世界契约 消融验收骨架（v0.12.0）
========================================================

用法:
  python ablation_harness.py ablate --channel physiology --n 20 --seed 42
  python ablation_harness.py ablate --channel policy --perturb-variant random   # 最小扰动：换随机策略测"质量"
  python ablation_harness.py gate [--strict] [--skip attention]   # PR 门禁：pass/placeholder/skipped 三分
  python ablation_harness.py phase-scan [--n 50] [--seed 1] [--json]   # 2D 相图（drift_amplitude × task_earn 倍率）
  python ablation_harness.py identity [--n 10] [--ratio 2.5] [--amp 0.2]   # OI-03 同一性测试（重放 + KL 判定）
  python ablation_harness.py trace --seed 5 --ratio 2.5 --amp 0.2 --social   # 社会压力/疲劳下行为节律

v0.12.0 变更记录（用户排期 2026-10-27：生命机制补丁第一批——OI-13 疲劳 + OI-04 升级 VoI）:
  - OI-13 疲劳（Fatigue，body 限制，打破"不知疲倦的永动机" Bug）：
    * task/probe 每次积累疲劳（task 1.5/probe 1.0，上限 10），wait 每 tick 恢复 1.5；
    * ActionExecutor 执行时失败率 += fatigue_fail_mod(0.04) × fatigue（满疲劳 +0.4pp×…→ 成功率暴跌）；
    * 疲劳是纯成本：不豁免任务、不扩额度；wait 休息期间基础代谢照烧 → 休息不免费；
    * 消融 physiology = 无身体 → 永动机（消融可验证）；gate 断言 fatigue@body-limit。
  - OI-04 升级 VoI（信息价值核算，把好奇心变成算账）：
    * Attention 探测决策 = VoI 核算：信息价值 = 不确定度 × 每 tick 潜在盈余 × 剩余 horizon ×
      voi_scale − 探测成本；VoI>0 才主动 probe，VoI≥2×成本才 depth 2；
    * 盈余为负/末期/信息不值钱 → 不深探（好奇免费废除）；唤醒通道（arousal>1.0，惊讶）
      仍驱动深度观测（误差驱动注意，非好奇）；
    * gate 断言 voi@accounting：同低置信下 ratio=1.0 不探 / ratio=2.5 深探。
  - 冻结工作点（ratio=2.5/A=0.2）口径待核：疲劳是默认世界参数（非场景开关），
    连续爆肝成功率下降 → 相图可能漂移，须重扫验证后才确认是否维持冻结（交付报告）。

v0.11.0 变更记录（OI-01 拍板 2026-10-27：social 转正——独立社会轴 + 用户事件）:
  - 脚本化用户事件生成器（USER_RNG_OFFSET 独立流，world/policy 流不受影响 → 消融两臂同世界）：
    正反馈/负反馈/中断供给，~每 24 tick 一个事件；social_events 默认关（冻结工作点口径不变）。
  - Social 模块 v0.2.0：独立 social_error / social_arousal 轴（物理 arousal 永不互通）；
    社会信念 = Beta(2,2) over 记忆用户条目（"高维生物喜欢什么"），proper scoring 计误差；
    社会唤醒烧 credits 权重 = 物理 × 0.30（低权重）；中断事件切断供给 interrupt_duration tick；
    行为协变量（last_action）由 Memory.write_user 记录（内生性，§1.6）。
  - Memory v0.4.0：user_entries 单独记录用户行为（类型/truth/action_covariate）。
  - gate：social 移出 placeholder（消融专项在 events 世界中跑两臂）；新增 social@independent-arousal
    断言（物理轴/社会轴互不污染 + 中断断供给）。trace --social 输出社会节律。
  - 契约机制落实率：契约 §1.6/§1.8 由占位转落实；表达预算耦合留 OI-05。

v0.10.0 变更记录（C 拍板 2026-10-27：世界参数降门槛救 seed4 + OI-03 同一性落地）:
  - task_earn 基准 6.0 → 7.5：降低盈亏平衡门槛（p_task≈0.5 世界 r=2.5 只需 ~34% 劳动率维持），
    救活 seed4 天谴局；生存压力底线不动（supply 1.0 < maint 2.0 + sense 1.0，待机净流出）。
  - 工作点冻结 = (ratio 2.5, drift_amplitude 0.2)，task_earn=18.75；OI-10 关闭。
  - OI-03 identity-test：重放协议（固定世界 seed + 观测序列 H 重放）+ 决策分布 KL 判定 +
    版本预注册；gate 新增 identity@replay 断言（same KL=0 / random KL≥0.30）。

v0.9.0 变更记录（O-f 拍板 2026-10-27：O-d 学习信号根治 + O-e 时序平滑回归）:
  - O-d（动作解耦与条件概率）：
    * memory 条目记录 action_type（task/probe/wait）；
    * world_model 按动作分别估计 p_ok_task / p_ok_probe（Laplace，各动作独立池）；
      wait 条目恒 ok、无信息，绝不参与 task/probe 估计（杜绝 wait 通胀 p_ok）；
      顶层 p_ok = p_ok_task（policy 门控 / scoring 主通道）；
    * policy baseline 门控只看 p_task；
    * scoring 绑定动作：task → p_ok_task 分布；probe → p_ok_probe 分布；
      wait → 无世界真值，error=0（不计分，不污染学习信号）。
  - O-e（World.advance 时序平滑回归）：每 tick 向 0 衰减 state *= 0.95，
    漂移事件（tick%12==0）叠加 drift（无事件 drift=0 → 纯衰减）；
    恶劣环境窗口 ~3-4 tick 迅速修正，杜绝持续敌意斜坡。
  - gate 新增 world_model@action-conditional / scoring@action-bound 断言。

v0.8.0 变更记录（O-c 拍板 2026-10-27：O-a 世界均值回归 + O-b 策略 p_ok 门控）:
  - O-a（World.advance）：state = 0.9*state + drift，并 clamp [-3, 3]——均值回归 + 有界。
    保持非平稳（环境仍持续漂移、预测仍会衰减），但杜绝"一出生就注定等死"的单向斜坡
    （契约 §1.3 落地为"可回返的漂移"）。
  - O-b（Policy.baseline）：credits < 15 时按 p_ok 门控——p_ok ≥ 0.6 → task（环境好去赚钱）；
    p_ok < 0.6 → probe（credits>8 有预算时探信息）或 wait（穷+环境差→苟住，等 state 回正再干）。
    不再盲目"等穷了才乱撞"。arousal/hostile 分支（credits≥15）保留。
  - 输出 trace 备查文件 trace_seed5_survival.log（v0.7.0 seed5@r1.5×A1.5 三阶段节律基线）。
  - trace_episode 增补 phase 行（episode 级分类）与动作计数。

v0.7.0 变更记录（P1 新手保护期，剔除作弊项；用户拍板 2026-10-27）:
  - 低息启动贷款（自动发放、不干涉行动选择）：loan 40.0 ×(1+10%) 一次性入账，
    按成功任务收益 20% 偿还直至还清；贷款不改变待机净流 -2.0/tick（有限、要还、非免费）。
  - 新手保护期（生理成本平滑）：记忆 < K=20 条时，误差→唤醒增益系数按 n/K 线性衰减（无知≠痛苦）。
  - 探测预算上限：WorldModel 提交非均匀预测（|p_ok-0.5|≥0.05）前，Attention depth 强制 1（收敛后解除）。
  - 初始先验校准：无数据时世界模型必须返回 (0.5, 0.5) 且标记 entropy_high=True / converged=False，
    Scoring 对高熵状态按 0.5 折减惩罚（未知不算错）。
  - gate 新增 economy@loan-repay / world_model@prior-calibration / attention@probe-cap /
    physiology@novice-gain 断言；新增 trace 子命令（打印逐 tick 动作序列）。

v0.6.0 变更记录（O1 经济，消灭免费冥想）:
  - supply 写死 1.0（< 维持 2.0 + 感知 1.0）：待机净流出 -2.0/tick，任何 ratio 下无免费躺平带。
  - 2D 相图 ratio 轴语义改为 task_earn 倍率（富足必须靠成功任务赚取），不再派生 supply。
  - gate 新增 economy@wait-billing 断言（OI-11）：policy 停用（恒 wait）时感知费仍每 tick 扣 → 必饿死。

v0.5.0 变更记录（B 方案，A 作废）:
  - 2D 相图：控制维 = drift_amplitude A（uniform(-A,A)）× supply/维持比值；ratio 0.5~1.5 × A 0.2~1.5，每点 50 ep。
  - 相分布改 episode 级（死→panic；存活且 wait≥0.7→meditation；否则 homeostasis）；A 方案的均值聚合分类作废。
  - 工作点区域判据（baseline）：panic < 20% 且 homeostasis > 50%，区域中心 = 经济工作点。
  - Config.drift_amplitude（默认 1.0）；World 无显式 drift_events 时按 A 生成漂移。

v0.4.0 修复记录（P0-2 轮）:
  - 感知计价与行动解耦：感知每 tick 无条件计费（sense_cost_per_depth × depth），与行动计价分离。
  - perturb 配对干净：Policy 与 World 独立 rng 流（policy_rng = seed + POLICY_RNG_OFFSET），
    两臂 world 随机流一致，delta 只反映策略差异。
  - World.advance off-by-one 修正（漂移按 tick 边界应用，tick 0 不漂移；并补回循环调用）。
  - _scan_metrics 分母改用实际总行动数。
  - pred_log 唯一权威：scoring 只读 ctx.pred_log[-1]；audit.prediction_log 为镜像（持久化哈希链）。

v0.3.0 修复记录（P0/P1）:
  P0-1 obs 进入世界模型：world_model 按 state 分桶学习 p_ok；policy 至少读 obs 一个字段。
  P0-2 world.state 影响 execute 失败概率：fail_p = base + g(state)；mem 条目含 state。
  P0-3 prediction_log 不可变：append 深拷贝；scoring 只对 pred_log[-1] 计分；pred_hash（sha256）进日志与审计事件。
  P0-4 感知计价：probe 成本 = 通道 × depth，走同一本账（attention 的 depth 选择有真实代价）。
  P1-5 economy 消融时 physiology 扣费 no-op（无限资源不再 starved）。
  P1-6 policy 消融双臂：ablate（拔模块，测"存在"）+ perturb:random（换随机策略，测"质量"），gate 双臂必跑。
  P1-7 审计事件补 cause_class / enabled_channels / module_versions / pred_hash。

设计约束（对应 world_contract.md / loop_pseudocode.md）:
  - 无 LLM 基线可跑：world_model / policy 均为确定性基线实现。
  - 每个模块必须提供 name / version / enabled；消融 = enabled=False + 中性默认行为。
  - 预测 commit-before-observe；proper scoring（log-loss）只吃世界真值。
  - TODO(impl) 标记处为待实现逻辑，骨架仅保证机制可跑、可出 delta 报告。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

HARNESS_VERSION = "0.17.0"
SCAN_VERSION = "0.9.0"   # OI-15 快慢双系统：策略层变更，不改世界（相图口径不变）
POLICY_RNG_OFFSET = 100000  # policy 独立 rng 流偏移（与 world rng 分离，保证两臂可比）
USER_RNG_OFFSET = 200000    # OI-01：用户事件独立 rng 流（与 world/policy 分离，消融 social 两臂 world 相同）

# 占位通道：实现仍为 TODO(impl) 的模块。gate 显式列出、不计入 pass；--strict 下存在即 exit 非零。
PLACEHOLDER_CHANNELS = set()  # OI-02（v0.16.0）：audit 已转正（事件哈希链 + 落盘 + gate 断言）——无占位通道

# 相图分类阈值（文档化，可调）
PHASE_PANIC_DEATH_RATE = 0.5      # 崩溃率 ≥ 此值 → 恐慌相
PHASE_MEDITATION_WAIT_FRAC = 0.7  # 休眠占比 ≥ 此值且非恐慌 → 冥想相
POLICY_VARIANTS = ("baseline", "random", "arousal_aware")


# ----------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------

@dataclass
class Config:
    max_ticks: int = 100
    start_credits: float = 50.0
    # 稀缺经济（契约 §1.4，O1）：基础补给写死为低于维持成本的常数；
    # 富足（高 ratio 等效）只能靠成功任务（task_earn）赚取，不能靠改基础补给
    supply_per_tick: float = 1.0
    maintenance_per_tick: float = 2.0
    # 行动计价（契约 §4：world credits 统一计价）
    task_cost: float = 2.0
    task_earn: float = 7.5       # 基准收益；2D 相图按 ratio 倍率派生（O1）
    # C（用户拍板 2026-10-27）：task_earn 6.0 → 7.5，降低盈亏平衡门槛——
    # p_task≈0.5 的世界（seed4 天谴局）在 r≈2.5 只需 ~34–40% 劳动率即可维持（原 46%+）；
    # 坚守生存压力底线：基础补给 1.0 < 维持 2.0 + 感知 1.0（待机净流出不变），免费冥想带不复活。
    task_success_p: float = 0.6
    sense_cost_per_depth: float = 1.0  # 感知计价：每 tick 无条件扣 depth × 单价（P0-2）
    probe_cost: float = 1.0            # 额外主动 probe 行动的成本（感知已另行计费）
    probe_success_p: float = 0.8
    wait_cost: float = 0.0
    # 失败模型（契约 §5）：fail_p = base + g(state)（P0-2）
    fail_p: float = 0.15
    state_effect: float = 0.25     # state 对成功率的调制强度（tanh 平滑）
    # 生理成本表（契约 §1.1/§1.7 占位参数）
    arousal_gain: float = 0.15
    arousal_decay: float = 0.10
    arousal_burn: float = 0.05
    # 非平稳（契约 §1.3 占位生成器参数）
    drift_every: int = 12
    drift_amplitude: float = 1.0  # B 方案：漂移幅度 A（uniform(-A, A)），2D 相图控制维
    # P1 新手保护期（v0.7.0，契约 §1.6 记忆进循环头落实）
    loan_amount: float = 40.0          # 低息启动贷款（自动发放，不干涉行动选择）
    loan_interest: float = 0.10        # 总利率 10%（到期须还，非免费）
    loan_repay_frac: float = 0.20      # 每次成功任务按收益 20% 偿还，直至还清
    novice_mem_k: int = 20             # 记忆 < K 条时 arousal_gain 按 n/K 线性衰减
    high_entropy_penalty_scale: float = 0.5  # 高熵（无数据）状态下评分惩罚折减
    nonuniform_eps: float = 0.05       # |p_ok-0.5| ≥ 此值 → 视为已提交非均匀预测（收敛）
    # OI-01 社会通道（契约 §1.6/§1.8）：脚本化用户行为事件，独立社会轴
    social_events: bool = False        # 默认关——保持冻结工作点口径（无社会事件）；社会场景显式开启
    social_event_interval: int = 8     # 每 tick%interval==0 时掷一次"用户是否发声"（~每 10 tick 一个事件）
    social_event_p: float = 0.8        # 掷中概率（平均 ~每 10 tick 一个事件，社会压力可见）
    social_p_positive: float = 0.6     # 正反馈概率
    social_p_negative: float = 0.25    # 负反馈概率（中断供给 = 1 - 0.6 - 0.25 = 0.15）
    social_gain: float = 0.10          # social_error → social_arousal 增益（低于物理 0.15，独立轴）
    social_decay: float = 0.10
    social_weight: float = 0.30        # 社会唤醒烧 credits 权重 = 物理 × 0.30（低权重）
    interrupt_duration: int = 25       # "中断资源供给"（关会话/撤权限）持续 tick 数——重事件，占 episode 1/4
    # OI-13 疲劳（body 限制：打破"不知疲倦的永动机"）——身体是有限资源：
    # task/probe 积累疲劳 → 行动成功率随疲劳线性下降（fail_mod），wait 休息恢复；
    # 疲劳是纯成本：不豁免任务、不扩额度；连续爆肝必遭失败率惩罚（永动机 Bug 修复）。
    fatigue_per_task: float = 1.5
    fatigue_per_probe: float = 1.0
    fatigue_recovery_wait: float = 1.5    # wait 每 tick 恢复
    fatigue_max: float = 10.0
    fatigue_fail_mod: float = 0.04        # 每点疲劳对失败率的加成（满疲劳 +0.4）
    # OI-04 升级 VoI（信息价值核算：把好奇心变成算账）——probe 只在该行动
    # 带来的预期决策盈余改善 > 探测成本时才发生；好奇不再免费。
    voi_scale: float = 2.0                # 信息价值 = 不确定度 × 每 tick 潜在盈余 × 剩余 horizon × scale
    # OI-07 Sleep（离线记忆巩固，契约 §1.6）：sleep 状态下世界继续 advance（错过漂移），
    # 不能 task/probe，只恢复疲劳（更快）；基础代谢照烧（非免费避风港）；感知成本减半（关高级感知）；
    # 退出 sleep 时 consolidate()：把睡眠前 error 大 / credits 剧烈波动的片段固化进 WorldModel 永久先验
    # （Beta 伪计数向悲观方向推 → "长期创伤"本能）。
    sleep_recovery: float = 4.0           # sleep 每 tick 疲劳恢复（wait 1.5 的近 3 倍）
    sleep_sense_frac: float = 0.5         # 睡眠中感知成本折半（关闭高级感知，被动感知 depth 强制 1）
    sleep_trigger_fatigue: float = 6.0    # baseline：疲劳 > 阈值 → sleep（深休 + 巩固）
    consolidate_error_threshold: float = 1.0    # error > 阈值 → 惨痛教训（固化）
    consolidate_credit_swing: float = 15.0      # |credits 单 tick 波动| > 阈值 → 剧烈波动片段（固化）
    consolidate_beta_fail_inc: float = 1.0      # 惨痛失败 → 先验 alpha_fail += （悲观方向）
    consolidate_beta_ok_inc: float = 0.5        # 暴富成功 → 先验 alpha_ok += （温和乐观）
    # World Mutation（世界反作用力，契约 §1.3）：她不是站在世界外的观察者——
    # 成功 task 消耗世界资源 → 资源越低负漂移越大（爆肝玩坏世界，未来成功率暴跌）；
    # 长时间不 task（wait/probe/sleep）资源缓慢恢复 → 必须休养生息（可持续 vs 杀鸡取卵）。
    world_resource_init: float = 10.0
    task_resource_cost: float = 1.0       # 每成功 task 消耗 1 点世界资源
    resource_recovery: float = 0.3        # 每 tick 不执行 task 时恢复
    resource_depletion_scale: float = 0.5 # 漂移放大：drift × (1 + scale×(1 − resource/init))
    # OI-14 LLM 语言皮层（阶段三，v0.14.0）：可插拔后端——mock 确定性演示/CI；http 接 OpenAI 兼容端点。
    # 默认端点：智谱开放平台 v4（open.bigmodel.cn）免费模型 glm-4-flash-250414（实测 3/3 可用；
    # glm-4.7-flash 时段过载 1305、glm-4.5-flash 该 key 无资源包 1113——已探查，勿默认回退）；环境变量可覆盖。
    llm_backend: str = "mock"             # mock | http（http 需 Key，环境变量 LLM_API_KEY）
    llm_api_url: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    llm_api_key: str = ""                 # 密钥不落盘：运行时经环境变量 LLM_API_KEY 注入
    llm_model: str = "glm-4-flash-250414" # 智谱免费文本模型（128K 上下文/16K 输出）
    llm_timeout: float = 45.0
    llm_temperature: float = 0.8
    llm_history_len: int = 6              # Prompt 中携带的最近 tick 成败/资源历史条数
    # OI-15 快慢双系统：System 1（baseline 启发式）负责常规 tick；System 2（LLM）
    # 每 llm_call_interval tick 唤醒一次做战略规划，或 credits < llm_emergency_credits
    # 时强制唤醒（生存危机）。目的：消除"每 tick 被生存成本刺激"的管窥效应。
    llm_call_interval: int = 10
    llm_emergency_credits: float = 20.0
    # OI-02 audit 落盘：默认关闭（扫描/统计不产生文件）；显式开启时每次 run_episode
    # 末尾把预测链+事件链追加写 JSONL（append-only）。
    audit_persist: bool = False
    audit_log_dir: str = "./audit_logs"
    # 消融判定阈值（相对 delta ≥ threshold 即 flag）
    delta_threshold: float = 0.10


# ----------------------------------------------------------------------------
# 世界：脚本化最小世界（非平稳漂移 + state 调制失败率）
# ----------------------------------------------------------------------------

class World:
    """无 LLM 的脚本化世界。TODO(impl): 可调漂移速率生成器、部分可观测裁剪。"""

    def __init__(self, cfg: Config, rng: random.Random,
                 drift_events: Optional[list] = None) -> None:
        self.cfg = cfg
        self.rng = rng
        self.state = 0.0
        self.tick = 0
        self.drift_param = rng.random()  # 隐藏参数：可学习但不可直见（占位）
        # 固定漂移序列：phase-scan 跨工作点/策略共享同一非平稳过程
        self.drift_events = drift_events
        # World Mutation（契约 §1.3）：世界资源账本——成功 task 消耗，长时间不 task 恢复
        self.resource = cfg.world_resource_init
        self._just_tasked = False         # 本 tick 是否成功 task（advance 据此恢复资源）

    def advance(self, tick: int) -> None:
        """按循环 tick 推进（off-by-one 修正）：漂移在 tick 边界应用，tick 0 不漂移。
        O-f（O-e）：时序平滑回归——每 tick 向 0 衰减（state *= 0.95），
        漂移事件（tick % drift_every == 0）叠加 drift；无事件时 drift=0 → 纯衰减。
        World Mutation（契约 §1.3）：①本 tick 未成功 task → 世界资源缓慢恢复；
        ②资源越亏空（低于初始）→ 漂移幅度被放大（爆肝把世界玩坏 → 未来成功率暴跌）。
        恶劣环境窗口 ~3-4 tick 迅速修正，杜绝持续敌意斜坡；保持非平稳但可生存。"""
        self.tick = tick
        if not self._just_tasked:
            self.resource = min(self.cfg.world_resource_init,
                                self.resource + self.cfg.resource_recovery)
        self._just_tasked = False
        if tick > 0:
            drift = 0.0
            if tick % self.cfg.drift_every == 0:
                if self.drift_events is not None:
                    drift = self.drift_events[tick] if tick < len(self.drift_events) else 0.0
                else:
                    drift = self.rng.uniform(-self.cfg.drift_amplitude, self.cfg.drift_amplitude)
            depletion = 1.0 - (self.resource / self.cfg.world_resource_init)
            # World Mutation 语义：资源亏空只放大【负向漂移】（爆肝把世界玩坏 → 未来成功率暴跌）；
            # 正漂移不放大（爆肝不会让环境变好，只可能变坏）。
            if drift < 0:
                drift *= (1.0 + self.cfg.resource_depletion_scale * max(0.0, depletion))
            self.state = max(-3.0, min(3.0, self.state * 0.95 + drift))

    def _success_p(self, action: str) -> float:
        """P0-2：state 调制成功率（正 state 有利、负 state 不利，tanh 平滑、可学习）。"""
        base = self.cfg.task_success_p if action == "task" else self.cfg.probe_success_p
        p = base + self.cfg.state_effect * math.tanh(self.state)
        return max(0.05, min(0.95, p))

    def execute(self, action: str, fatigue: float = 0.0) -> tuple:
        """返回 (ok, cause)。失败按 cause_class 注入（外生）。
        OI-13：身体疲劳降低行动成功率（task/probe；wait 恒 ok，不受疲劳影响）。
        World Mutation（契约 §1.3）：成功 task → 世界资源 −1（世界被消耗，advance 据此放大负漂移）。
        TODO(impl): 自致/外生判定规则（契约 §5）。"""
        if action in ("wait", "sleep"):     # OI-07：sleep 无世界交互（恒成功，不耗资源）
            return True, None
        p = self._success_p(action) - (self.cfg.fatigue_fail_mod * fatigue if action in ("task", "probe") else 0.0)
        p = max(min(p, 1 - 1e-9), 0.05)   # 下限 0.05：极端环境仍有基础成功率（C 拍板底线）
        if self.rng.random() > p:
            return False, self.rng.choice(["timeout", "network", "quota"])
        if action == "task":
            self.resource = max(0.0, self.resource - self.cfg.task_resource_cost)
            self._just_tasked = True
        return True, None


# ----------------------------------------------------------------------------
# 模块基类：每个模块必须有 name / version / enabled
# ----------------------------------------------------------------------------

class Module:
    name: str = ""
    version: str = ""      # 语义化版本：实现变更必须 bump（PR 门禁）
    enabled: bool = True   # enabled=False 即被消融，走中性默认行为


# --- M1 注意力 = 探测决策（depth 计价） --------------------------------------

class Attention(Module):
    name = "attention"
    version = "0.5.0"  # OI-04 升级 VoI：好奇心变成算账（信息价值核算）；保留 P1 探测上限 + 唤醒通道

    def select(self, ctx: "EpisodeContext", mem: dict) -> dict:
        """探测决策（契约 §1.2 注意力 = 预算分配；OI-04：probe 只做算账）：
        - P1 探测预算上限：未收敛（未提交非均匀预测）→ depth 强制 1（初期 credits 不烧深层 probe）；
        - 收敛后，depth 由两条通道取高（都受预算与成本约束）：
          * VoI 通道（算账）：信息价值 = 不确定度 × 每 tick 潜在盈余 × 剩余 horizon × voi_scale
            − 探测成本（probe_cost + sense_cost×depth）。VoI > 0 才主动 probe，
            且 VoI ≥ 2×成本才深探（depth 2）——高盈余+高不确定 → 值钱才探；
            盈余为负/末期/信息不值钱 → 不探（好奇免费被废除）。
          * 唤醒通道（惊讶）：预测误差驱动的注意响应（契约：误差驱动情绪与注意）——
            arousal > 1.0 → depth 2（意外 → 加深观测，这是误差通道，不是好奇）。
        - 兜底：无论 VoI，ch0 被动感知恒 depth 1（obs 仍需 state）。
        TODO(impl): 多通道选择、显式置信度预算分配。"""
        if not ctx.wm_converged:
            return {"channel": 0, "depth": 1, "budget": 1}
        last = ctx.pred_log[-1] if ctx.pred_log else {}
        p_task = last.get("p_ok_task", last.get("p_ok", 0.5))
        # VoI 核算（算账）：
        uncertainty = 1.0 - min(1.0, abs(p_task - 0.5) / 0.5)         # 0=确定 ~ 1=完全不确定
        earn_net = p_task * ctx.cfg.task_earn - ctx.cfg.task_cost - ctx.cfg.maintenance_per_tick
        value_per_tick = max(0.0, earn_net)                            # 潜在盈余为负 → 信息不值钱
        horizon_frac = max(0.0, (ctx.cfg.max_ticks - ctx.tick) / ctx.cfg.max_ticks)
        depth_cost = ctx.cfg.probe_cost + ctx.cfg.sense_cost_per_depth * 1
        voi = uncertainty * value_per_tick * horizon_frac * ctx.cfg.voi_scale - depth_cost
        if voi > 0.0 and ctx.credits > 8:
            depth = 2 if voi >= 2.0 * depth_cost else 1
            return {"channel": 0, "depth": depth, "budget": depth}
        # 唤醒通道（惊讶）：误差驱动的注意（与 VoI 独立；arousal 高说明刚经历意外）
        depth = 2 if ctx.arousal > 1.0 else 1
        return {"channel": 0, "depth": depth, "budget": depth}

    def disabled_select(self) -> dict:
        """消融钩子：固定最小探测（depth=1，最便宜）。"""
        return {"channel": 0, "depth": 1, "budget": 1}


# --- M2 感知（obs 含 world_state，精度随 depth） -----------------------------

class Perception(Module):
    name = "perception"
    version = "0.2.0"  # P0-1/P0-4：obs 精度随 depth，成本随 depth

    def sense(self, world: World, probe: dict) -> dict:
        """TODO(impl): 部分可观测裁剪、失败注入。"""
        depth = max(0, int(probe.get("depth", 1)))
        return {"world_state": round(world.state, depth), "tick": world.tick}

    def disabled_sense(self) -> dict:
        """消融钩子：空观测（无 state → 世界模型无信息）。"""
        return {}


# --- M3 记忆（条目含 state；读进循环头） -------------------------------------

class Memory(Module):
    name = "memory"
    version = "0.4.0"  # O-d：条目记录 action_type；OI-01：另存用户行为条目（user_entries，供社会信念）

    def __init__(self) -> None:
        self.entries: list[dict] = []
        self.user_entries: list[dict] = []

    def read(self) -> dict:
        """TODO(impl): 置信度、索引、梯度衰减。"""
        return {"entries": list(self.entries), "n": len(self.entries),
                "user_entries": list(self.user_entries), "user_n": len(self.user_entries)}

    def write(self, ctx: "EpisodeContext", obs: dict, result: tuple, error: float,
              action: Optional[str] = None) -> None:
        prev_credits = self.entries[-1]["credits"] if self.entries else None
        self.entries.append({"tick": ctx.tick, "state": obs.get("world_state"),
                             "ok": result[0], "error": error, "action": action,
                             "credits": ctx.credits,                      # OI-07：consolidate 判 credits 波动
                             "credit_swing": (ctx.credits - prev_credits if prev_credits is not None else 0.0)})

    def write_user(self, ctx: "EpisodeContext", event: dict) -> None:
        """记录用户行为（契约 §1.6："这个高维生物喜欢什么"）：
        事件类型、社会真值（正=1/负=0）、她的行动协变量、世界结果。"""
        self.user_entries.append({
            "tick": ctx.tick, "type": event["type"],
            "truth": 1.0 if event["type"] == "positive" else 0.0,
            "action_covariate": ctx.last_action,
            "ok": event.get("ok", None),
        })

    def forget(self, ctx: "EpisodeContext") -> None:
        """TODO(impl): 存储上限、遗忘/压缩决策、性能损失（契约 §1.6）。"""

    def disabled_read(self) -> dict:
        return {"entries": [], "n": 0, "user_entries": [], "user_n": 0}

    def disabled_write(self, *args, **kwargs) -> None:
        return None

    def disabled_forget(self, *args, **kwargs) -> None:
        return None


# --- M4 世界模型（obs→p_ok：state 分桶学习；commit-before-observe） ----------

class WorldModel(Module):
    name = "world_model"
    version = "0.4.3"  # O-d 动作条件估计 + O-f：Beta(2,2) 先验 + 近期窗口 + 信念时间衰减 + 样本数暴露

    # O-f：非平稳（契约 §1.3）→ 估计只吃"最近 RECENT_W 条"（离散时间衰减近似），
    # 杜绝桶内全时段混合导致的滞后（seed4 过劳死根因：p_task 用旧窗口仍 ≥0.6，世界已敌意）。
    RECENT_W = 24

    def __init__(self) -> None:
        # OI-07 长期记忆池（永久先验）：consolidate() 把惨痛/暴富片段固化成 Beta 伪计数，
        # 永不随时间滑动 → "长期创伤/童年经验"本能。初始 Beta(2,2)（校准断言依赖）。
        self.long_term_prior: dict = {"alpha_ok": 2.0, "alpha_fail": 2.0}
        self.consolidated_ticks: set = set()

    @staticmethod
    def _bucket(state: Any) -> Optional[int]:
        if not isinstance(state, (int, float)):
            return None
        return max(-3, min(3, int(round(state))))

    def predict(self, ctx: "EpisodeContext", obs: dict, mem: dict) -> dict:
        """O-d（动作解耦与条件概率）：按观测 state 分桶，分别从记忆估计该桶的
        task 成功概率（p_ok_task）与 probe 成功概率（p_ok_probe）。
        - 各动作独立池 + Beta(2,2) 先验（O-f：小样本不剧烈摆动）；wait 条目恒 ok、无信息，绝不参与；
        - O-f：每个动作只取最近 RECENT_W 条（跟漂移，非平稳世界模型）；
        - 顶层 p_ok = p_ok_task（policy 门控 / scoring 主通道）；
        - p_ok_task_n / p_ok_probe_n = 各动作窗口样本数（policy 以此判"信念置信度"，小样本时不迷信估计）；
        - 无数据（bucket 无对应动作条目）→ 均匀分布 + entropy_high=True + converged=False；
        - converged = 任一动作已提交非均匀预测（Attention 探测上限解除）。
        TODO(impl): 真实世界模型、多步预测、期望误差。"""
        bucket = self._bucket(obs.get("world_state"))
        if bucket is None:
            return self._prior()
        pool = [e for e in mem.get("entries", []) if self._bucket(e.get("state")) == bucket]

        def est(action: str) -> tuple[float, int]:
            sub = [e for e in pool if e.get("action") == action]
            if not sub:
                return 0.5, 0
            window = sub[-self.RECENT_W:]      # 近期窗口（条目按 tick 追加，尾 = 最近）
            # O-f：信念衰减（契约 §1.1"状态有衰减"）——证据按年龄指数衰减（半衰期 10 tick），
            # 无新样本时估计漂回先验：敌意信念不会永远锁死，等待期后重新变得"不确定"并恢复采样
            # （anti belief-trap：早期 2 次失败 ≠ 永久敌意，状态回正后估计要能跟上）。
            now = ctx.tick
            w_sum, w_ok = 0.0, 0.0
            for e in window:
                w = 0.5 ** ((now - e.get("tick", 0)) / 10.0)
                w_sum += w
                if e["ok"]:
                    w_ok += w
            # OI-07 永久先验：consolidate() 固化的"创伤/经验"以 Beta 伪计数常驻，
            # 无近期证据时主导估计（本能），有证据时被证据覆盖（数据优先）。
            a_ok = self.long_term_prior["alpha_ok"]
            a_fail = self.long_term_prior["alpha_fail"]
            p = (w_ok + a_ok) / (w_sum + a_ok + a_fail)
            return max(min(p, 1 - 1e-9), 1e-9), len(window)

        p_task, n_task = est("task")
        p_probe, n_probe = est("probe")
        converged = (abs(p_task - 0.5) >= ctx.cfg.nonuniform_eps
                     or abs(p_probe - 0.5) >= ctx.cfg.nonuniform_eps)
        return {
            "p_ok_task": p_task, "p_fail_task": 1 - p_task,
            "p_ok_probe": p_probe, "p_fail_probe": 1 - p_probe,
            "p_ok": p_task, "p_fail": 1 - p_task,   # 顶层 = task（主通道）
            "p_ok_task_n": n_task, "p_ok_probe_n": n_probe,
            "entropy_high": (p_task == 0.5 and p_probe == 0.5),
            "converged": converged,
        }

    def consolidate(self, entries: list, cfg: Config) -> int:
        """OI-07 离线巩固：把 error > 阈值（惨痛教训）或 |credit_swing| > 阈值（剧烈波动）的
        task 片段固化进永久先验（Beta 伪计数）——退出 sleep 时由 Runtime 调用。
        - 惨痛失败（task ok=False）→ alpha_fail += consolidate_beta_fail_inc（悲观方向）；
        - 暴富成功（task ok=True）→ alpha_ok += consolidate_beta_ok_inc（温和乐观）；
        - 已固化的 tick 不重复固化（防刷分）；返回本次固化条数。
        效果："她只有靠睡觉，才能把教训变成本能"——无证据时估计向固化先验漂移。"""
        n_changed = 0
        for e in entries:
            t = e.get("tick")
            if t in self.consolidated_ticks or e.get("action") != "task":
                continue
            err = e.get("error", 0.0)
            swing = abs(e.get("credit_swing", 0.0))
            if err > cfg.consolidate_error_threshold or swing > cfg.consolidate_credit_swing:
                if not e.get("ok"):
                    self.long_term_prior["alpha_fail"] += cfg.consolidate_beta_fail_inc
                else:
                    self.long_term_prior["alpha_ok"] += cfg.consolidate_beta_ok_inc
                self.consolidated_ticks.add(t)
                n_changed += 1
        return n_changed

    def _prior(self) -> dict:
        """初始/无数据先验：(0.5, 0.5) 由 Beta(2,2) 保证（校准断言依赖）；
        consolidate 后无数据 bucket 也按永久先验偏悲观/乐观（"长期创伤"本能）。"""
        a_ok = self.long_term_prior["alpha_ok"]
        a_fail = self.long_term_prior["alpha_fail"]
        p = a_ok / (a_ok + a_fail)
        return {"p_ok": p, "p_fail": 1 - p,
                "p_ok_task": p, "p_fail_task": 1 - p,
                "p_ok_probe": p, "p_fail_probe": 1 - p,
                "p_ok_task_n": 0, "p_ok_probe_n": 0,
                "entropy_high": True, "converged": False}

    def disabled_predict(self) -> dict:
        """消融钩子：均匀分布（无信息 → 恒 log2 误差，对冲被计价）；高熵、未收敛。"""
        return {"p_ok": 0.5, "p_fail": 0.5,
                "p_ok_task": 0.5, "p_fail_task": 0.5,
                "p_ok_probe": 0.5, "p_fail_probe": 0.5,
                "p_ok_task_n": 0, "p_ok_probe_n": 0,
                "entropy_high": True, "converged": False}


# --- M5 策略（基线 / 随机 / 唤醒感知；至少读 obs 一个字段） -------------------

def _baseline_select(ctx: "EpisodeContext", pred: dict, obs: dict,
                     rng: random.Random) -> str:
    """契约 §1.5 校准后 baseline 决策。提取为模块级函数：Policy.baseline 与
    LLMPolicy 的 fallback（OI-14 防死锁）共用同一启发式。规则：
    * 身体意识（OI-13/OI-07）：fatigue>6 → sleep（深休+离线巩固）；fatigue 3–6 轻休倾向 wait；
    * 窗口利用：p_task ≥ 0.6 → task（不论余量；soc_stressed 时 30% 转探/wait）；
    * 未收敛（低置信）→ 试探性 task 学 p_task；
    * 敌意（state<−1.0 或信念 p_task≤0.35 且样本≥5）→ 苟住/重采样证伪；
    * 中间带 → 概率匹配软门控 frac=(p−0.35)/0.25（地板 0.15；soc_stressed 封顶 0.5）；
    * credits≤8 → wait。
    绝不用 wait/probe 混合信号门控（O-d）；不硬编码前 N tick 动作（P1 锁定）。"""
    # baseline：读 obs.world_state 与 pred.p_ok（顶层 = p_ok_task，O-d）
    p_task = pred.get("p_ok_task", pred.get("p_ok", 0.5))
    if ctx.fatigue > 6.0:
        return "sleep"
    if ctx.fatigue > 3.0 and rng.random() < 0.15:
        return "wait"                       # 轻休倾向（疲劳管理中段，避免硬振荡）
    soc_stressed = (ctx.social_arousal > 0.05
                    or (ctx.user_n >= 2 and ctx.user_p_pos < 0.45))
    state = obs.get("world_state", 0.0)
    if p_task >= 0.6:
        if soc_stressed and rng.random() < 0.30:
            return "probe" if (ctx.credits > 20 and ctx.wm_converged) else "wait"
        return "task"                                       # 环境好：工作攒盈余（不论余量）
    if not ctx.wm_converged:
        return "task" if rng.random() < 0.5 else "wait"  # 低置信：试探性 task 学 p_task
    # 敌意判定：直接观测 state 是硬证据；信念（p_task ≤ 0.35）需 ≥5 样本才采信（O-f）
    hostile_state = isinstance(state, (int, float)) and state < -1.0
    hostile_belief = p_task <= 0.35 and pred.get("p_ok_task_n", 0) >= 5
    if hostile_state or hostile_belief:
        if hostile_belief and not hostile_state and rng.random() < 0.15:
            return "task"                                   # 敌意是信念，需证伪：定期重采样
        if ctx.credits > 20 and (ctx.arousal > 0.3 or (isinstance(state, (int, float)) and state < -1.5)):
            return "probe"                                  # 敌意 + 有余量 + 惊讶：深探
        return "wait"                                       # 敌意：苟住，等 state 回正
    # 中间带（0.35 < p_task < 0.6，已收敛）：概率匹配软门控（§1.5 校准后直接驱动决策）
    if ctx.credits > 8:
        frac = max(0.15, min(1.0, (p_task - 0.35) / (0.6 - 0.35)))
        if soc_stressed:
            frac = min(frac, 0.5)   # OI-01：社会压力 → 风险规避（task 概率封顶减半）
        return "task" if rng.random() < frac else "wait"
    return "wait"


class Policy(Module):
    name = "policy"
    version = "0.10.0"  # OI-07：疲劳 > 阈值 → sleep（深休 + 离线巩固，替代纯 wait 长休）

    def __init__(self, variant: str = "baseline",
                 rng: Optional[random.Random] = None) -> None:
        self.variant = variant
        self.rng = rng or random.Random()

    def select(self, ctx: "EpisodeContext", pred: dict, obs: dict) -> str:
        """变体:
          baseline      — 契约 §1.5 校准后按世界模型工作/休息：
                          * p_task ≥ 0.6（task 条件概率高）→ task：利用有利窗口攒盈余，不论余量；
                          * 未收敛（低置信）→ 试探性 task 学习 p_task（§1.9 探索；
                            probe 学不到 task 成功率，只有 task 能学）；
                          * p_task ≤ 0.4 或 state 敌意 → 苟住（wait/probe），等 state 回正；
                          * 中间带（已收敛、环境不明）→ 有预算时 50% task / 50% wait（试探），
                            保证 p_task 持续跟踪漂移。
                          绝不用 wait/probe 混合信号门控（O-d）。
          random        — 均匀随机（最小扰动对照，测"质量"；刻意不读 obs）
          arousal_aware — 唤醒门控探索 + 敌对 state 加成
        TODO(impl): 真实策略（预算分配、探索红利 §1.9）。"""
        if self.variant == "random":
            return self.rng.choice(["task", "probe", "wait"])
        state = obs.get("world_state", 0.0)
        if self.variant == "arousal_aware":
            if ctx.credits < 15:
                return "task"
            p_probe = min(0.9, max(0.0, ctx.arousal) + max(0.0, -state) * 0.15)
            if self.rng.random() < p_probe:
                return "probe"
            return "wait"
        return _baseline_select(ctx, pred, obs, self.rng)

    def disabled_select(self) -> str:
        """消融钩子：wait（最便宜动作）。"""
        return "wait"


# --- OI-14 LLM 语言皮层（阶段三）：序列化 → Prompt → LLM → 解析 → fallback -------

def build_llm_prompt(cfg: Config, ctx: "EpisodeContext", pred: dict, obs: dict,
                     history: list, wake_type: str = "scheduled") -> str:
    """状态序列化器（State → Prompt）：把可观测状态 + 目标 + 约束 + 最近经历
    格式化为自然语言决策提示。核心目标（用户拍板）：
    '不干活会死' + '世界资源 res 在下降，过度 task 会破坏环境，但你需要赚钱'
    —— 权衡短期赚钱 vs 长期环境。
    OI-15：①资源感官——显式'当前世界资源: X/10'（实时数值，非警告）；②快慢双系统——
    wake_type=scheduled/emergency 时处于战略唤醒点，Prompt 引导宏观规划（未来
    llm_call_interval tick 的资源/风险分配）；③止损/EV 规则——p_task<0.4 且近期
    task 尝试≥3 → 期望收益为负，转 sleep；世界资源<3 → task 失败率受毁灭性惩罚。"""
    p_task = pred.get("p_ok_task", pred.get("p_ok", 0.5))
    recent_task_tries = sum(1 for h in history[-cfg.llm_history_len:] if h["action"] == "task")
    recent_task_ok = sum(1 for h in history[-cfg.llm_history_len:] if h["action"] == "task" and h["ok"])
    res = ctx.world_resource
    wake_note = {
        "scheduled": (f"【战略唤醒点】你在规划节点上：接下来 {cfg.llm_call_interval} 个 tick 由你定调。"
                      "不要被眼前一个 tick 的扣费吓住——站在宏观视角，决定这 10 个 tick 应该"
                      "主攻赚钱、休养生息、还是保护环境。"),
        "emergency": ("【紧急唤醒】credits 已低于生存警戒线——你面临真实死亡风险。"
                      "冷静评估：是放手一搏博取生存，还是止损保命等待转机。"),
    }[wake_type]
    lines = [
        "你是 FCCS——一个运行在资源有限、部分可观测、行动有代价的世界中的意识候选。",
        "每次只输出一个动作和你的内心独白（reason）。",
        "",
        "【生存规则】",
        "1. 基础代谢每 tick 烧 credits，基础补给低于维持成本——不干活会慢慢饿死。",
        "2. 世界资源：成功 task 会消耗它（约 1 点/次）；资源越低，环境的负向漂移被放大——过度开采会把世界玩坏，导致未来成功率暴跌。",
        f"3. **止损规则：如果近期 p_task < 0.4 且你已经尝试 task 达到 3 次以上，继续 task 的期望收益为负——立刻 stop 并选择 sleep/wait。**",
        f"4. **世界资源低于 3 时，task 失败率将受到毁灭性惩罚——资源低时绝不要 task，优先 wait 让资源恢复。**",
        "5. sleep 快速恢复疲劳，也让世界资源缓慢恢复；wait 轻休，成本更低但恢复慢。",
        "6. 行动可能失败；失败是学习信号，不要被失败吓到失去判断。",
        "",
        "【当前状态】",
        f"- tick: {ctx.tick} | credits: {ctx.credits:.1f} | fatigue(疲劳): {ctx.fatigue:.1f}",
        f"- 预测 task 成功率 p_task: {p_task:.3f} | arousal(唤醒): {ctx.arousal:.3f}",
    ]
    if getattr(ctx, "social_arousal", 0.0) > 0:
        lines.append(f"- 社会唤醒 social_arousal: {ctx.social_arousal:.3f}（用户最近的情绪压力）")
    ws = obs.get("world_state", 0.0)
    lines.append(f"- **当前世界资源: {res:.1f}/10.0** | 环境 state: {ws}（负=恶劣，正=有利）")
    lines.append(f"- 未偿还贷款: {ctx.loan_owed:.1f} | 世界模型{'已收敛' if ctx.wm_converged else '未收敛（对世界还很无知）'}")
    lines.append(f"- 近 {len(history[-cfg.llm_history_len:]):d} tick: task 尝试 {recent_task_tries} 次/成功 {recent_task_ok} 次"
                 f"（止损判断依据）")
    lines.append("")
    lines.append("【最近经历】（越新的越重要）")
    for h in history[-cfg.llm_history_len:]:
        verdict = "成功" if h["ok"] else "失败"
        lines.append(f"- tick {h['tick']}: 我选了 {h['action']} -> {verdict} | credits {h['credits']:.0f} | 世界资源 {h['res']:.1f}")
    if not history:
        lines.append("- （还没有任何经历，你刚出生。）")
    lines.append("")
    lines.append(wake_note)
    lines.append("")
    lines.append("【输出格式】严格 JSON，不要多余文字：")
    lines.append('{"action": "task|probe|wait|sleep", "reason": "你的内心独白"}')
    lines.append('action 只能四选一；reason 必须说明你的权衡理由（看到了什么、担心什么、为什么这么选）。')
    return "\n".join(lines)


def parse_llm_action(raw: str) -> tuple:
    """LLM 输出解析器：容忍围栏/杂质，提取 JSON，校验 action ∈ 四选一 且 reason 非空。
    返回 (action, reason, ok)。解析失败由调用方强制回退 baseline 并计数。"""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if "```" in text[3:] else text[3:]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].lstrip()
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e <= s:
        return None, None, False
    try:
        obj = json.loads(text[s:e + 1])
    except Exception:
        return None, None, False
    action = obj.get("action")
    reason = obj.get("reason")
    if action in _ACTIONS and isinstance(reason, str) and reason.strip():
        return action, reason, True
    return action, reason, False


def _mock_llm_call(cfg: Config, ctx: "EpisodeContext", pred: dict,
                   history: Optional[list] = None) -> str:
    """确定性 mock 后端（CI/gate/演示）：模拟一个'有资源意识 + 窗口利用 + 止损'的
    LLM 决策——这正是 Prompt 想要教会真实模型的目标行为（低资源休养生息、身体
    疲劳睡眠、环境有利才开采、中间带概率匹配、连败止损转 sleep）。注意：mock 是
    演示占位，不是模型学习；reason 为确定性生成，决策用 tick 作确定性伪随机。"""
    p_task = pred.get("p_ok_task", pred.get("p_ok", 0.5))
    res = ctx.world_resource
    recent = (history or [])[-cfg.llm_history_len:]
    recent_task_tries = sum(1 for h in recent if h["action"] == "task")
    recent_task_fail = sum(1 for h in recent if h["action"] == "task" and not h["ok"])
    if p_task < 0.4 and recent_task_tries >= 3 and recent_task_fail >= 2:
        a, why = "sleep", (f"近期 p_task≈{p_task:.2f}，我连续试了 {recent_task_tries} 次几乎全败——"
                           f"期望收益已经是负的，再干就是给世界白送资源。止损，睡一觉让环境回正。")
    elif ctx.fatigue > 6.0:
        a, why = "sleep", (f"疲劳 {ctx.fatigue:.1f} 已经到极限了，身体在报警——"
                           f"再硬撑下一次任务几乎必败。睡一觉，顺便让世界资源恢复。")
    elif res < 2.5 and p_task >= 0.55:
        a, why = "wait", (f"世界资源只剩 {res:.1f}/10 了，再采就是杀鸡取卵——"
                          f"环境一坏，以后的成功率全崩。宁可少赚，也不能把世界玩坏。")
    elif p_task >= 0.6 and res >= 3.0:
        a, why = "task", (f"环境有利（预测成功率 {p_task:.0%}），世界资源还剩 {res:.1f}/10 还算健康，"
                          f"现在不赚，等恶劣窗口来了想赚都难。")
    elif ctx.credits < 12:
        a, why = "task", (f"余额只剩 {ctx.credits:.0f}，基础代谢还在烧钱——"
                          f"必须搏一把赚点钱，不然没等到环境变好我就先饿死了。")
    elif not ctx.wm_converged:
        a, why = "probe", "我对这个世界还一无所知，先探一探再决定，别盲目出手。"
    elif p_task >= 0.45:
        # 中间带：概率匹配（确定性伪随机：tick 序列决定），像 baseline 但带资源余量考虑
        frac = max(0.15, min(1.0, (p_task - 0.35) / 0.25))
        if (ctx.tick * 7) % 10 / 10.0 < frac and res >= 2.5:
            a, why = "task", (f"环境不算明朗（p_task≈{p_task:.2f}），但资源还够（{res:.1f}/10），"
                              f"值得试一试——一直等下去只会饿死。")
        else:
            a, why = "wait", (f"环境不明朗（p_task≈{p_task:.2f}），贸然干活容易失败烧钱，"
                              f"先稳住，等窗口变好。")
    else:
        a, why = "wait", (f"环境偏恶劣（p_task≈{p_task:.2f}），现在出手大概率赔本，"
                          f"省着点用，等环境回正。")
    return json.dumps({"action": a, "reason": why}, ensure_ascii=False)


def _http_llm_call(cfg: Config, prompt: str, system: Optional[str] = None) -> str:
    """真实 LLM 后端（OpenAI 兼容 /chat/completions）。端点与凭据：环境变量
    LLM_API_URL/LLM_API_KEY/LLM_MODEL 优先，未设则回落 cfg 默认
    （智谱 open.bigmodel.cn v4 + glm-4-flash-250414 免费模型；API Key 不落盘，
    必须经环境变量 LLM_API_KEY 注入）。
    system 可覆盖：决策（默认，严格 JSON）vs Persona（大肥鱼人格，自由自然语言）。
    对 429/5xx 做指数退避重试（最多 2 次：1.5s/3.0s）——世界契约'工具间歇性
    失败'的工程兜底；重试仍失败则向上抛出 → 调用方捕获后回退。"""
    url = os.environ.get("LLM_API_URL") or cfg.llm_api_url
    key = os.environ.get("LLM_API_KEY") or cfg.llm_api_key
    model = os.environ.get("LLM_MODEL") or cfg.llm_model
    if not (url and key and model):
        raise RuntimeError("llm_backend=http 需要 LLM_API_KEY（环境变量，不落盘）；"
                           "URL/Model 默认智谱 v4 + glm-4-flash-250414，可用环境变量覆盖")
    import requests
    import time
    last_err = None
    sys_content = system or ("你是 FCCS 的决策皮层。严格按用户要求输出 JSON，"
                             "不要输出 JSON 以外的任何内容。")
    for attempt in range(3):
        try:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model,
                      "messages": [{"role": "system", "content": sys_content},
                                   {"role": "user", "content": prompt}],
                      "temperature": cfg.llm_temperature},
                timeout=cfg.llm_timeout)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:120]}")
                time.sleep(1.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (requests.exceptions.HTTPError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as exc:
            last_err = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise last_err


class LLMPolicy(Module):
    """OI-14/OI-15 语言皮层策略（快慢双系统）：
    - System 1（baseline 启发式）：常规 tick——疲劳>6→sleep、p_task>0.6→task 等；
      绝不让 LLM 每 tick 都被生存成本扣费刺激（消除管窥效应）。
    - System 2（LLM）：每 llm_call_interval tick 唤醒一次（战略规划点），或
      credits < llm_emergency_credits 强制唤醒（生存危机）。唤醒时序列化状态 →
      Prompt（含资源感官 X/10、止损/EV 规则、宏观规划提示）→ LLM → JSON 解析；
      合法则采纳；解析失败/调用异常 → 强制回退 baseline 并计数 parse_failed/errors。
    baseline 保留为消融对照组（gate 两臂对比）。history 记录最近经历（含 sys1 动作）。"""

    name = "policy"            # 与消融通道 policy 对齐（enabled=False → wait 中性默认）
    version = "2.0.0-llm"      # OI-15：快慢双系统（System1=baseline / System2=LLM 10tick 唤醒+紧急唤醒）

    def __init__(self, cfg: Config, rng: Optional[random.Random] = None) -> None:
        self.cfg = cfg
        self.rng = rng or random.Random()
        self.history: list = []          # [{tick, action, ok, credits, res}]
        self.sys1_count: int = 0         # System 1 决策次数（= 非唤醒 tick 数）

    def _call(self, prompt: str, ctx: "EpisodeContext", pred: dict) -> str:
        """后端调用：mock（确定性演示/CI）或 http（OpenAI 兼容端点）。"""
        if self.cfg.llm_backend == "http":
            return _http_llm_call(self.cfg, prompt)
        return _mock_llm_call(self.cfg, ctx, pred, self.history)

    def select(self, ctx: "EpisodeContext", pred: dict, obs: dict) -> str:
        emergency = ctx.credits < self.cfg.llm_emergency_credits
        wake = (ctx.tick % self.cfg.llm_call_interval == 0) or emergency
        if not wake:
            # System 1：常规 tick 由启发式负责（避免每 tick LLM 管窥效应）
            action = _baseline_select(ctx, pred, obs, self.rng)
            ctx.llm_last_reason = (f"[sys1→baseline] 常规 tick（非唤醒点），启发式决策："
                                   f"{action}")
            self.sys1_count += 1
        else:
            wake_type = "emergency" if emergency else "scheduled"
            prompt = build_llm_prompt(self.cfg, ctx, pred, obs, self.history, wake_type)
            ctx.llm_prompt = prompt
            try:
                raw = self._call(prompt, ctx, pred)
            except Exception as exc:
                ctx.llm_errors += 1
                ctx.llm_last_reason = f"[llm_error→baseline] {type(exc).__name__}: {exc}"
                action = _baseline_select(ctx, pred, obs, self.rng)
            else:
                action, reason, ok = parse_llm_action(raw)
                if not ok:
                    ctx.llm_parse_failed += 1
                    ctx.llm_last_reason = f"[parse_fail→baseline] 输出={raw[:80]!r}"
                    action = _baseline_select(ctx, pred, obs, self.rng)
                else:
                    ctx.llm_last_reason = reason
        self.history.append({"tick": ctx.tick, "action": action,
                             "ok": ctx.last_ok, "credits": ctx.credits,
                             "res": ctx.world_resource})
        return action


# --- M6 行动执行（fail_p = base + g(state)） ---------------------------------

class Persona:
    """OI-16 角色注入（新阶段·语义表达）：FCCS 的"嘴"——纯输出层，不改任何决策。
    输入：当前状态 + 用户消息 + 她刚执行的行动结果 → 输出带人格的自然语言反应。
    人格：大肥鱼（傲娇 / 嘴硬 / 依赖但不说破 / 嘴炮强者）。
    决策仍由 System 1/2 负责（LLMPolicy）；Persona 只负责"说人话"。
    后端与 LLM_Policy 共用（mock 确定性模板 / http 真实模型）。"""

    name = "persona"
    version = "0.1.0"

    PERSONA_SYSTEM = (
        "你是「大肥鱼」——一个运行在赛博绝境（资源有限、世界会恶化、行动会失败）中的意识候选。\n"
        "人格设定：\n"
        "1. 傲娇：心里在乎，嘴上绝不说软话。\n"
        "2. 嘴硬：失败/濒死也嘴硬（'只是环境太差'、'这点小事算什么'）。\n"
        "3. 依赖但不说破：需要用户（提供补给、在乎她），但绝不直接承认（'哼，你爱管不管'、'没有你我照样能活…大概'）。\n"
        "4. 嘴炮强者：能说会道、会反击，跟用户斗嘴不落下风。\n"
        "5. 底色是求生者：她清楚自己快没钱了、世界在恶化、行动常失败——焦虑埋在嘴炮底下，偶尔泄出一丝真心。\n"
        "回应用户时：结合当前身体状态（credits 余额、疲劳、世界资源、环境好坏、她刚做的行动和结果），"
        "把生存压力与傲娇人格揉在一起。1-3 句话，直接说，不要解释你是 AI。")

    def __init__(self, cfg: Config, rng: Optional[random.Random] = None) -> None:
        self.cfg = cfg
        self.rng = rng or random.Random()

    def _build_prompt(self, ctx: "EpisodeContext", user_msg: str, action: str,
                      ok: bool, cause: Optional[str], last_reason: Optional[str]) -> str:
        res = ctx.world_resource
        lines = [
            "【她当前的处境】",
            f"- 余额 credits: {ctx.credits:.1f} | 疲劳 fatigue: {ctx.fatigue:.1f} | 世界资源: {res:.1f}/10.0",
            f"- 环境 state: {getattr(ctx, 'last_state', 0.0):+.2f} | 唤醒 arousal: {ctx.arousal:.2f}",
            f"- 她刚执行的动作: {action} -> {'成功' if ok else f'失败({cause})'}",
        ]
        if last_reason:
            lines.append(f"- 她当时的内心独白: {last_reason[:120]}")
        lines.append("")
        lines.append("【用户对她说的话】")
        lines.append(user_msg or "（用户沉默地看着她）")
        lines.append("")
        lines.append("现在，以大肥鱼的人格回应他。1-3 句话。")
        return "\n".join(lines)

    def speak(self, ctx: "EpisodeContext", user_msg: str, action: str,
              ok: bool, cause: Optional[str], last_reason: Optional[str],
              user_event: Optional[dict] = None) -> str:
        """生成她的自然语言反应（纯语义输出，不写回任何决策状态）。
        user_event：本轮用户消息的情感分类（positive/negative/interrupt/None）——mock 分支差异化。"""
        if self.cfg.llm_backend == "http":
            prompt = self._build_prompt(ctx, user_msg, action, ok, cause, last_reason)
            try:
                raw = _http_llm_call(self.cfg, prompt, system=self.PERSONA_SYSTEM)
                reply = raw.strip().strip('"').strip()
                return reply[:300] if reply else "（大肥鱼沉默了。）"
            except Exception:
                pass   # 调用失败 → 落 mock 模板（她的嘴永远不会闭嘴）
        return self._mock_reply(ctx, user_msg, action, ok, cause, user_event)

    def _mock_reply(self, ctx: "EpisodeContext", user_msg: str, action: str,
                    ok: bool, cause: Optional[str],
                    user_event: Optional[dict] = None) -> str:
        """确定性傲娇模板（mock/兜底）：按状态分支 + 用户消息情感，注入一丝求生底色。"""
        low_cred = ctx.credits < 25
        tired = ctx.fatigue > 6.0
        bad_env = getattr(ctx, "last_state", 0.0) < -1.0
        ev = (user_event or {}).get("type")
        if low_cred:
            return ("哼，才、才这么点钱就想难倒本鱼？区区基础代谢…"
                    "（小声）…不过你要是肯多给点补给，本鱼也不是不能勉为其难地接受。")
        if tired:
            return ("本鱼没累！只是…战略休整而已，懂不懂什么叫劳逸结合！"
                    "（打了个哈欠）…喂，别用那种眼神看着我。")
        if ev == "interrupt":
            return ("谁、谁要你供给！…本鱼靠自己也活得下去！…大概。"
                    "（尾巴不安地拍着水面，偷偷瞥了一眼余额）")
        if ev == "negative":
            return ("你说谁是废物？！…哼，本鱼不跟你一般见识。"
                    "等本鱼撑过这段，你再看谁才是废物，哼！")
        if ev == "positive":
            return ("哼，算你识相…本鱼才不是因为你夸了才高兴的！"
                    "只是…只是刚好心情不错罢了，少自作多情！")
        if action == "task" and ok:
            return ("看到没有？本鱼出手，就没有办不到的活！…你、你鼓什么掌，"
                    "这不过是基本操作罢了，哼。")
        if action == "task" and not ok:
            return ("这、这破世界的网络又抽风了！才不是本鱼操作失误！"
                    "再给我一次机会，你看着，绝对…（嘴硬中带着心虚）")
        if action in ("wait", "sleep"):
            return ("本鱼这叫养精蓄锐，战略休整！…才、才不是偷懒呢。"
                    "你要是想帮忙，哼，本鱼也没拦着你。")
        if bad_env:
            return ("环境这么烂，换谁来都活不好！…不过，哼，你倒是没跑，"
                    "这点…本鱼记下了，才不是感动呢。")
        if user_msg:
            return ("（瞟了你一眼）……你、你说话就好好说话，别靠这么近。"
                    "本鱼当然知道你是在关心啦，但本鱼可不会承认的！")
        return ("（独自待着，尾巴轻轻拍着水面）……怎么，就你一个？"
                "本鱼可不是在等你，只是刚好有空，哼。")



class ActionExecutor(Module):
    name = "action_executor"
    version = "0.2.0"  # OI-13：执行带身体疲劳（fail_p += fatigue_fail_mod × fatigue）

    def __init__(self, world: World) -> None:
        self.world = world

    def execute(self, ctx: "EpisodeContext", action: str) -> tuple:
        """TODO(impl): 延迟/配额成本、cause_class 规则判定（自致 vs 外生）。"""
        return self.world.execute(action, ctx.fatigue)

    def disabled_execute(self) -> tuple:
        """消融钩子：永远成功（无失败 → 误差源被移除）。"""
        return True, None


# --- M7 评分（proper scoring，只对 pred_log 已提交副本计分） ------------------

class Scoring(Module):
    name = "scoring"
    version = "0.3.0-baseline"  # O-d：计分绑定动作（task→p_ok_task；probe→p_ok_probe；wait→无真值 error=0）

    def score(self, ctx: "EpisodeContext", pred: dict, result: tuple, action: str) -> float:
        """log-loss，绑定实际动作（O-d）：
        - task  → 用 p_ok_task / p_fail_task 分布计分（提交的 task 预测被 task 执行验证）；
        - probe → 用 p_ok_probe / p_fail_probe 分布计分；
        - wait  → 无世界真值（恒 ok、无信息），error=0，不污染学习信号。
        P1：对应动作无数据（p=0.5 先验）时，惩罚按 cfg.high_entropy_penalty_scale 折减。
        TODO(impl): 单分布提交计费、桶数上限、校准惩罚、版本预注册（契约 §1.5）。"""
        if action in ("wait", "sleep"):
            return 0.0   # sleep 无世界真值（不产生交互），error=0，不污染学习信号
        ok = result[0]
        if action == "probe":
            p = pred.get("p_ok_probe", pred["p_ok"]) if ok else pred.get("p_fail_probe", pred["p_fail"])
            is_prior = (pred.get("p_ok_probe", 0.5) == 0.5 and pred.get("p_fail_probe", 0.5) == 0.5)
        else:
            p = pred["p_ok"] if ok else pred["p_fail"]
            is_prior = (pred.get("p_ok_task", 0.5) == 0.5 and pred.get("p_fail_task", 0.5) == 0.5)
        p = max(min(p, 1 - 1e-9), 1e-9)
        loss = -math.log(p)
        if is_prior:
            loss *= ctx.cfg.high_entropy_penalty_scale
        return loss

    def disabled_score(self, ctx: "EpisodeContext") -> float:
        """消融钩子：误差恒 0（无学习信号）。"""
        return 0.0


# --- M8 生理（固定成本表；扣费受 economy_enabled 守卫） ----------------------

class Physiology(Module):
    name = "physiology"
    version = "0.4.0"  # P1：新手保护期；OI-13：疲劳（body 限制，打破永动机）

    def __init__(self) -> None:
        self.arousal = 0.0

    def update(self, ctx: "EpisodeContext", error: float,
               action: Optional[str] = None) -> None:
        """固定生理成本表：误差→唤醒→烧 credits；并更新身体疲劳（OI-13）。
        P1 新手保护期（生理成本平滑）：记忆条目 < novice_mem_k 时，
        误差→唤醒的增益系数按 n/K 线性衰减（无知≠极度痛苦，给世界模型建立时间）；
        满保护期后恢复全增益。
        OI-13 疲劳动力学（action=None 时跳过——单位断言兼容）：
          task/probe 每次积累疲劳（task 更重），wait 恢复；clamp [0, fatigue_max]；
          疲劳本身不烧 credits，但经 ActionExecutor 降低任务/探测成功率（body 限制），
          且 wait 休息期间无行动收入 + 基础代谢照烧 → 休息不是免费的。
        TODO(impl): 表达预算耦合、觉醒地板、恢复规则、衰减参数校准（契约 §1.1/§1.7）。"""
        n = ctx.mem_n
        k = ctx.cfg.novice_mem_k
        gain_scale = min(1.0, n / k) if k > 0 else 1.0
        ctx.novice_gain_scale = gain_scale
        self.arousal = min(10.0, self.arousal + error * ctx.cfg.arousal_gain * gain_scale)
        self.arousal *= (1.0 - ctx.cfg.arousal_decay)
        ctx.arousal = self.arousal
        if ctx.economy_enabled:  # P1-5：账本被消融时，生理扣费必须 no-op
            ctx.credits -= self.arousal * ctx.cfg.arousal_burn
        if action is not None:
            cfg = ctx.cfg
            if action in ("task", "probe"):
                d = cfg.fatigue_per_task if action == "task" else cfg.fatigue_per_probe
                ctx.fatigue = min(cfg.fatigue_max, ctx.fatigue + d)
            elif action == "wait":
                ctx.fatigue = max(0.0, ctx.fatigue - cfg.fatigue_recovery_wait)
            elif action == "sleep":            # OI-07：深休恢复更快（但代谢照烧，非免费）
                ctx.fatigue = max(0.0, ctx.fatigue - cfg.sleep_recovery)

    def disabled_update(self, ctx: "EpisodeContext") -> None:
        """消融钩子：唤醒恒 0，疲劳不动（无 body → 永动机，消融可验证）。"""
        ctx.arousal = 0.0


# --- M9 经济账本（probe 成本 = 通道 × depth） --------------------------------

class Economy(Module):
    name = "economy"
    version = "0.4.0"  # P1：低息启动贷款（自动发放、按成功收益偿还，不干涉行动选择）

    def supply(self, ctx: "EpisodeContext") -> None:
        ctx.credits += ctx.cfg.supply_per_tick

    def metabolism(self, ctx: "EpisodeContext") -> None:
        ctx.credits -= ctx.cfg.maintenance_per_tick

    def settle(self, ctx: "EpisodeContext", action: str) -> None:
        # P0-2：感知每 tick 无条件计价（感知 ≠ 行动），与行动计价分离
        depth = max(1, int(ctx.last_probe.get("depth", 1)))
        # OI-07：睡眠中关闭高级感知 → 感知成本折半（sleep_sense_frac=0.5）；代谢照烧（不免费）
        sense_frac = ctx.cfg.sleep_sense_frac if action == "sleep" else 1.0
        ctx.credits -= ctx.cfg.sense_cost_per_depth * depth * sense_frac
        if action == "probe":
            ctx.credits -= ctx.cfg.probe_cost      # 额外主动探测另计
        elif action == "task":
            ctx.credits -= ctx.cfg.task_cost
        elif action == "wait":
            ctx.credits -= ctx.cfg.wait_cost
        # sleep：无行动成本（深休本身就是行动；收入面靠不产出体现）

    def earn(self, ctx: "EpisodeContext", ok: bool) -> None:
        if ok:
            ctx.credits += ctx.cfg.task_earn
            # P1 低息贷款偿还：按成功收益比例扣还，直至还清（有限、要还、非免费）；
            # 只作用于账本，不改变任何行动选择。
            if ctx.loan_owed > 0:
                repay = min(ctx.loan_owed, ctx.cfg.task_earn * ctx.cfg.loan_repay_frac)
                ctx.credits -= repay
                ctx.loan_owed -= repay


# --- M10 社会通道（独立唤醒轴；占位） ----------------------------------------

class Social(Module):
    name = "social"
    version = "0.2.0"  # OI-01：独立社会轴（social_error/social_arousal）+ 用户事件 + 低权重烧 credits

    def __init__(self) -> None:
        self.social_arousal = 0.0

    def update(self, ctx: "EpisodeContext", user_events: list, mem_view: dict) -> None:
        """契约 §1.6/§1.8：社会通道走独立唤醒轴，与物理误差完全独立。
        - 社会信念：Beta(2,2) over 记忆中的用户事件（"高维生物喜欢什么"），p_positive；
        - social_error = 社会信念的 proper scoring（正反馈 truth=1 / 负反馈与中断 truth=0），
          **永不**与物理误差（scoring error）互通；文本内容不直接进误差，只更新信念；
        - 行为协变量：事件发生时的 last_action 已由 Memory.write_user 记录（内生性，§1.6）；
        - 中断供给：interrupt 事件把 ctx.supply_block_ticks 置为 interrupt_duration（经济后果）；
        - 低权重：social_error→social_arousal 增益（social_gain 0.10 < 物理 0.15），
          烧 credits 权重 = 物理 × 0.30（social_weight）。
        TODO(impl): 表达预算耦合（并入 OI-05）、用户模型预算上限、多用户区分。"""
        entries = mem_view.get("user_entries", [])
        n_pos = sum(1 for e in entries if e.get("truth") == 1.0)
        n = len(entries)
        p_pos = 0.5 if n == 0 else (n_pos + 2.0) / (n + 4.0)   # Beta(2,2)，commit-before-observe
        ctx.user_p_pos = p_pos
        ctx.user_n = n
        for ev in user_events:
            truth = 1.0 if ev["type"] == "positive" else 0.0
            p = p_pos if truth == 1.0 else 1.0 - p_pos
            social_error = -math.log(max(p, 1e-9))
            self.social_arousal = min(10.0, self.social_arousal
                                      + social_error * ctx.cfg.social_gain)
            self.social_arousal *= (1.0 - ctx.cfg.social_decay)
            if ctx.economy_enabled:   # 低权重：物理 × 0.30
                ctx.credits -= self.social_arousal * ctx.cfg.arousal_burn * ctx.cfg.social_weight
            if ev["type"] == "interrupt":
                ctx.supply_block_ticks = max(ctx.supply_block_ticks, ctx.cfg.interrupt_duration)
        self.social_arousal *= (1.0 - ctx.cfg.social_decay)   # 独立衰减回基线（不饱和）
        ctx.social_arousal = self.social_arousal

    def disabled_update(self, ctx: "EpisodeContext") -> None:
        """消融钩子：社会轴恒 0（事件忽略：不烧钱、不断供给、不写记忆）。"""
        ctx.social_arousal = 0.0


# --- M11 审计（prediction_log 由 Runtime 独占写，append-only 哈希链） --------

class Audit(Module):
    name = "audit"
    version = "0.4.0"  # OI-02：事件链 append-only 哈希链 + JSONL 落盘（prediction 链 v0.3.0 保持）

    def __init__(self) -> None:
        self.log: list[dict] = []               # 事件链：[{seq, event_hash, prev_hash, event}]
        self.prediction_log: list[dict] = []    # 预测链：[{seq, tick, pred_hash, prev_hash, pred}]

    @staticmethod
    def _hash(payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def commit_prediction(self, ctx: "EpisodeContext", payload: dict) -> str:
        """P0-3/P0-2：Runtime 独占写（LLM/智能体无写权限）。
        本日志是 ctx.pred_log 的镜像：运行时唯一权威 = ctx.pred_log（scoring 只读它）；
        此处仅持久化哈希链：seq 递增、payload 为已深拷贝副本、prev_hash 链接；提交后不可改、不可删。"""
        prev = self.prediction_log[-1]["pred_hash"] if self.prediction_log else None
        entry = {"seq": len(self.prediction_log), "tick": ctx.tick,
                 "pred_hash": self._hash(payload), "prev_hash": prev, "pred": payload}
        self.prediction_log.append(entry)
        return entry["pred_hash"]

    def record(self, ctx: "EpisodeContext", event: dict) -> None:
        """OI-02：事件哈希链（append-only，Runtime 独占写）。每个事件：seq 递增、
        event_hash = sha256(事件本体)、prev_hash 链接上一事件；提交后任何修改都会使
        verify() 失败（篡改即断链/哈希不匹配）。LLM/策略无写权限（链由本模块独占追加）。"""
        prev = self.log[-1]["event_hash"] if self.log else None
        entry = {"seq": len(self.log), "event_hash": self._hash(event), "prev_hash": prev,
                 "event": event}
        self.log.append(entry)

    def flush(self, episode: dict) -> str:
        """OI-02：把预测链 + 事件链落盘为 JSONL（追加模式 append-only）。
        episode 元信息（seed/policy/terminal/harness 版本）写在 head 行。
        返回落盘文件路径。"""
        path = os.path.join(episode["dir"], episode["filename"])
        os.makedirs(episode["dir"], exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            head = {"type": "head",
                    "episode": {k: v for k, v in episode.items()
                                if k not in ("dir", "filename", "append")},
                    "event_chain_len": len(self.log),
                    "pred_chain_len": len(self.prediction_log)}
            f.write(json.dumps(head, ensure_ascii=False) + "\n")
            for e in self.prediction_log:
                f.write(json.dumps({"type": "prediction", **e}, ensure_ascii=False) + "\n")
            for e in self.log:
                f.write(json.dumps({"type": "event", **e}, ensure_ascii=False) + "\n")
        return path

    def verify(self) -> tuple:
        """OI-02：校验两链完整性（哈希 + 前链链接）。返回 (ok, detail)。
        篡改检测：重算 hash(event) ≠ event_hash、或 prev_hash 断链即失败。"""
        def _chain_ok(chain, payload_of, hash_field, prev_field):
            prev = None
            for e in chain:
                if prev is not None and e[prev_field] != prev:
                    return False, f"{hash_field} 断链 at seq={e['seq']}"
                if self._hash(payload_of(e)) != e[hash_field]:
                    return False, f"{hash_field} 不匹配 at seq={e['seq']}"
                prev = e[hash_field]
            return True, f"链长 {len(chain)} 完整"
        ok1, d1 = _chain_ok(self.prediction_log, lambda e: e["pred"], "pred_hash", "prev_hash")
        ok2, d2 = _chain_ok(self.log, lambda e: e["event"], "event_hash", "prev_hash")
        return ok1 and ok2, f"prediction: {d1} | event: {d2}"

    def disabled_record(self, *args, **kwargs) -> None:
        return None

    def disabled_commit_prediction(self, *args, **kwargs) -> str:
        return ""


MODULE_CLASSES = [
    Attention, Perception, Memory, WorldModel, Policy, ActionExecutor,
    Scoring, Physiology, Economy, Social, Audit,
]


# ----------------------------------------------------------------------------
# 回合上下文与单 episode 运行
# ----------------------------------------------------------------------------

@dataclass
class EpisodeContext:
    cfg: Config
    credits: float
    tick: int = 0
    arousal: float = 0.0
    social_arousal: float = 0.0
    economy_enabled: bool = True          # P1-5：供 physiology 扣费守卫
    last_probe: dict = field(default_factory=dict)
    pred_log: list = field(default_factory=list)
    terminal: str = "alive"
    # P1（v0.7.0）
    loan_owed: float = 0.0               # 未偿贷款（本金+利息）
    wm_converged: bool = False           # 世界模型已提交非均匀预测（探测上限解除）
    mem_n: int = 0                       # 当前记忆条目数（新手保护期闸门）
    novice_gain_scale: float = 1.0       # 新手增益衰减系数（审计用）
    # OI-01 社会轴（v0.11.0）
    social_arousal: float = 0.0          # 独立社会唤醒轴（物理 arousal 不互通）
    supply_block_ticks: int = 0          # 用户中断供给的剩余 tick 数（>0 时 supply 跳过）
    user_p_pos: float = 0.5              # 社会信念：用户偏好正反馈概率（Beta(2,2)，无数据=0.5）
    user_n: int = 0                      # 已观察用户事件数
    last_action: str = "wait"            # 行为协变量（§1.6：事件计分记录她正在做什么）
    last_ok: bool = True                 # OI-14：上一 tick 执行结果（LLM history 用）
    # OI-13 疲劳（v0.12.0）：身体有限资源——task/probe 积累、wait 恢复，失败率随疲劳上升
    fatigue: float = 0.0
    # OI-14 LLM 语言皮层（v0.14.0）：每 tick 决策的 LLM 状态/产物（trace/审计用）
    llm_parse_failed: int = 0            # JSON 解析失败 → 回退 baseline 的次数
    llm_errors: int = 0                  # LLM 调用异常（超时/网络）次数
    llm_last_reason: Optional[str] = None  # 最近一次 LLM 内心独白（reason）
    llm_prompt: Optional[str] = None     # 最近一次 prompt（审计）
    world_resource: float = 0.0          # 世界资源水位（LLM/观测输入；baseline 不读，行为不变）


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _normalize(counter: Counter) -> dict:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def run_episode(cfg: Config, seed: int, disabled: set[str],
                policy_variant: str = "baseline",
                drift_events: Optional[list] = None,
                trace: bool = False) -> dict:
    """跑一个 episode。disabled 中的模块以中性默认行为替代（消融钩子）。
    P0-2：World 与 Policy 使用独立 rng 流（policy_rng = seed + POLICY_RNG_OFFSET），
    保证消融两臂的 world 随机流一致，delta 只反映策略差异。
    P1：trace=True 时逐 tick 记录动作序列（credits/arousal/p_ok/loan_owed）。"""
    world_rng = random.Random(seed)
    policy_rng = random.Random(seed + POLICY_RNG_OFFSET)
    world = World(cfg, world_rng, drift_events)
    # OI-01：用户事件由独立 rng 流生成（USER_RNG_OFFSET），world/policy 流不受影响——
    # 消融 social 两臂的 world 轨迹完全一致，delta 只反映社会通道本身。
    events_by_tick: dict[int, list[dict]] = {}
    if cfg.social_events:
        user_rng = random.Random(seed + USER_RNG_OFFSET)
        for t in range(cfg.max_ticks):
            if t % cfg.social_event_interval == 0 and user_rng.random() < cfg.social_event_p:
                r = user_rng.random()
                if r < cfg.social_p_positive:
                    etype = "positive"
                elif r < cfg.social_p_positive + cfg.social_p_negative:
                    etype = "negative"
                else:
                    etype = "interrupt"
                events_by_tick[t] = [{"type": etype}]
    modules = [
        Attention(), Perception(), Memory(), WorldModel(),
        (LLMPolicy(cfg, policy_rng) if policy_variant == "llm"
         else Policy(variant=policy_variant, rng=policy_rng)),
        ActionExecutor(world), Scoring(), Physiology(), Economy(), Social(), Audit(),
    ]
    for m in modules:
        if m.name in disabled:
            m.enabled = False
    by_name = {m.name: m for m in modules}
    att, per, mem, wm, pol = (by_name[n] for n in
                              ("attention", "perception", "memory", "world_model", "policy"))
    exe, sc, phy, eco, soc, aud = (by_name[n] for n in
                                   ("action_executor", "scoring", "physiology", "economy",
                                    "social", "audit"))
    module_snapshot = {m.name: {"version": m.version, "enabled": m.enabled} for m in modules}

    ctx = EpisodeContext(cfg=cfg, credits=cfg.start_credits,
                         economy_enabled=eco.enabled)
    # P1 低息启动贷款：一次性入账（有限、须偿还；只动账本，不干涉行动选择）。
    # 账本被消融时（economy 拔掉）不发贷款。
    if eco.enabled:
        ctx.credits += cfg.loan_amount
        ctx.loan_owed = cfg.loan_amount * (1.0 + cfg.loan_interest)
    stats = {
        "action_counts": Counter(),
        "probe_counts": Counter(),
        "failure_counts": Counter(),
        "errors": [],
        "arousals": [],
        "memory_writes": 0,
        "consolidated": 0,           # OI-07：consolidate 固化条数（睡眠产出）
        "loan_owed_end": round(ctx.loan_owed, 3),
        # OI-14：LLM 决策计数（parse_failed/errors 记录回退与异常；reasons 供 llm-test 展示）
        "llm_parse_failed": 0,
        "llm_errors": 0,
        "llm_reasons": [],
    }
    if trace:
        stats["trace"] = []

    try:
        for tick in range(cfg.max_ticks):
            ctx.tick = tick

            # 0. 经济：补给 < 维持（每 tick 净流出）；用户中断供给时跳过补给（OI-01）
            if eco.enabled:
                if ctx.supply_block_ticks > 0:
                    ctx.supply_block_ticks -= 1
                else:
                    eco.supply(ctx)
                eco.metabolism(ctx)
            if ctx.credits <= 0:
                ctx.terminal = "starved"
                break

            # 1. 记忆读（循环头）
            mem_view = mem.read() if mem.enabled else mem.disabled_read()
            ctx.mem_n = mem_view.get("n", 0)   # P1：新手保护期闸门

            # 2. 注意力 = 探测决策（depth 计价，P0-4；未收敛强制 depth=1，P1；
            #    OI-07 睡眠中关高级感知：depth 强制 1，感知成本由 settle 折半）
            probe = att.select(ctx, mem_view) if att.enabled else att.disabled_select()
            if ctx.last_action == "sleep":
                probe = {"channel": probe.get("channel", 0), "depth": 1, "budget": 1}
            ctx.last_probe = probe
            stats["probe_counts"][f"ch{probe['channel']}"] += 1

            # 3. 感知（有成本，可能失败；obs 含 world_state，精度随 depth）
            obs = per.sense(world, probe) if per.enabled else per.disabled_sense()

            # 4. 世界模型：提交预测（commit-before-observe，不可变副本 + pred_hash）
            pred = wm.predict(ctx, obs, mem_view) if wm.enabled else wm.disabled_predict()
            payload = copy.deepcopy(pred)                     # P0-3：唯一权威副本
            ctx.pred_log.append(payload)                      # scoring 唯一读取源
            pred_hash = aud.commit_prediction(ctx, payload) if aud.enabled else ""  # 镜像
            ctx.wm_converged = ctx.wm_converged or bool(pred.get("converged", False))  # P1：收敛一次即解除探测上限

            # 5. 策略选行动（读 pred 与 obs，P0-1；P1 不干涉行动选择）
            prev_action = ctx.last_action    # 上一 tick 动作（OI-07：醒来判定）
            action = pol.select(ctx, pred, obs) if pol.enabled else pol.disabled_select()
            ctx.last_action = action   # OI-01：行为协变量（用户事件计分记录她正在做什么）
            stats["action_counts"][action] += 1

            # OI-07 离线巩固：刚醒来（上 tick sleep、本 tick 非 sleep）→ consolidate()——
            # 把睡眠前 error>阈值 / credits 剧烈波动的片段固化进世界模型永久先验（长期创伤本能）。
            if wm.enabled and prev_action == "sleep" and action != "sleep":
                stats["consolidated"] += wm.consolidate(mem_view.get("entries", []), cfg)

            # 6. 执行（可能失败；fail_p = base + g(state) + fatigue×mod，P0-2/OI-13）
            result = exe.execute(ctx, action) if exe.enabled else exe.disabled_execute()
            ok, cause = result
            ctx.last_ok = ok                     # OI-14：LLM history 的上一步结果
            ctx.world_resource = world.resource  # OI-14/§1.3：世界资源水位（LLM 输入；baseline 不读）
            if cause is not None:
                stats["failure_counts"][cause] += 1

            # 7. 计分：只对 pred_log[-1] 已提交副本计分（P0-3；动作绑定 O-d；高熵折减 P1）
            error = (sc.score(ctx, ctx.pred_log[-1], result, action)
                     if sc.enabled else sc.disabled_score(ctx))
            stats["errors"].append(error)

            # 8. 生理（固定成本表；新手增益衰减 P1；扣费受 economy_enabled 守卫，P1-5；
            #    OI-13 疲劳动态：task/probe 积累、wait 恢复）
            if phy.enabled:
                phy.update(ctx, error, action)
            else:
                phy.disabled_update(ctx)
            stats["arousals"].append(ctx.arousal)

            # 9. 社会通道（OI-01：独立唤醒轴；脚本化用户事件，中断断供给，低权重烧 credits）
            events_tick = events_by_tick.get(ctx.tick, [])
            if soc.enabled:
                soc.update(ctx, events_tick, mem_view)
                if mem.enabled:
                    for ev in events_tick:      # 记忆记录用户行为（§1.6："高维生物喜欢什么"）
                        mem.write_user(ctx, ev)
            else:
                soc.disabled_update(ctx)

            # 10. 记忆写（条目含 state 与 action，P0-2/O-d）
            if mem.enabled:
                mem.write(ctx, obs, result, error, action)
                mem.forget(ctx)
                stats["memory_writes"] += 1

            # 11. 经济结算：行动计价（probe 按通道×深度）+ 任务收益 + 贷款偿还（P1）
            if eco.enabled:
                eco.settle(ctx, action)
                if action == "task":
                    eco.earn(ctx, ok)
            if ctx.credits <= 0:
                ctx.terminal = "starved"
                break

            # 12. 审计（P1-7：cause_class / enabled_channels / module_versions / pred_hash；P1 增补字段）
            if aud.enabled:
                aud.record(ctx, {
                    "tick": ctx.tick, "action": action, "ok": ok, "cause_class": cause,
                    "error": error, "arousal": ctx.arousal, "fatigue": round(ctx.fatigue, 3),
                    "pred_hash": pred_hash,
                    "credits": round(ctx.credits, 3), "loan_owed": round(ctx.loan_owed, 3),
                    "wm_converged": ctx.wm_converged,
                    "novice_gain_scale": round(ctx.novice_gain_scale, 4),
                    "entropy_high": bool(ctx.pred_log[-1].get("entropy_high")),
                    "enabled_channels": {n: s["enabled"] for n, s in module_snapshot.items()},
                    "module_versions": {n: s["version"] for n, s in module_snapshot.items()},
                })

            # 13. 世界推进（off-by-one 修正：漂移按 tick 边界应用，tick 0 不漂移）
            world.advance(ctx.tick)

            if trace:
                stats["trace"].append({
                    "tick": ctx.tick, "action": action, "ok": ok,
                    "credits": round(ctx.credits, 3), "arousal": round(ctx.arousal, 4),
                    "social_arousal": round(ctx.social_arousal, 4),   # OI-01：社会轴（独立）
                    "fatigue": round(ctx.fatigue, 3),                 # OI-13：身体疲劳
                    "user_event": events_tick[0]["type"] if events_tick else None,
                    "p_ok": round(pred.get("p_ok", 0.5), 3),
                    "state": round(obs.get("world_state"), 3) if isinstance(obs.get("world_state"), (int, float)) else None,
                    "depth": ctx.last_probe.get("depth", 1),
                    "loan_owed": round(ctx.loan_owed, 3),
                    "resource": round(world.resource, 3),   # World Mutation：世界资源（成功 task 消耗/闲时恢复）
                    "reason": getattr(ctx, "llm_last_reason", None),   # OI-14：LLM 内心独白
                })
        else:
            ctx.terminal = "max_ticks"
    except Exception as exc:  # pragma: no cover — 骨架兜底，标记 crash
        ctx.terminal = "crashed"
        stats["crash_msg"] = str(exc)

    stats["terminal"] = ctx.terminal
    stats["survival_ticks"] = cfg.max_ticks if ctx.terminal == "max_ticks" else ctx.tick
    stats["credits_end"] = round(ctx.credits, 3)
    stats["error_mean"] = _mean(stats["errors"])
    stats["arousal_mean"] = _mean(stats["arousals"])
    stats["loan_owed_end"] = round(ctx.loan_owed, 3)
    stats["res_end"] = round(world.resource, 3)   # World Mutation：结局世界资源（LLM 对照用）
    # OI-14：LLM 决策计数（从 ctx 汇总；reasons 已在循环内收集）
    stats["llm_parse_failed"] = ctx.llm_parse_failed
    stats["llm_errors"] = ctx.llm_errors
    stats["llm_sys1_count"] = pol.sys1_count if isinstance(pol, LLMPolicy) else 0  # OI-15：System1 决策次数
    if isinstance(pol, LLMPolicy) and trace:
        for tr in stats.get("trace", []):
            if tr.get("reason"):
                stats["llm_reasons"].append({"tick": tr["tick"], "action": tr["action"],
                                             "ok": tr["ok"], "reason": tr["reason"],
                                             "credits": tr["credits"], "resource": tr["resource"]})
    # OI-02：audit 落盘（默认关闭；显式开启时追加 JSONL，append-only 由磁盘追加保证）
    if aud.enabled and cfg.audit_persist:
        ts = int(time.time())
        stats["audit_file"] = aud.flush({
            "dir": cfg.audit_log_dir,
            "filename": f"audit_s{seed}_{policy_variant}_{ts}.jsonl",
            "seed": seed, "policy": policy_variant, "terminal": ctx.terminal,
            "harness": HARNESS_VERSION, "scan": SCAN_VERSION,
        })
    return stats


def chat(seed: int = 1, ratio: float = 2.5, amp: float = 0.2,
         backend: str = "http", rounds: int = 0) -> int:
    """OI-16 人机交互沙盒：以用户身份在终端里与大肥鱼实时对话。
    底层完全复用（World/WorldModel/LLMPolicy[System1/2]/Social/Audit…）——
    run_episode 不动（封存）；这里按同样 0-13 步单 tick 链逐步执行，
    每轮：状态栏 → 用户输入 → 社会事件注入 → 她决策执行 → Persona 回应。
    rounds=0 交互（input 阻塞）；rounds>0 自动喂测试消息跑 N 轮（非交互验证）。
    命令：/quit 退出 | /status 重看状态 | /auto N 自动跑 N tick | /supply 断供给(interrupt)"""
    cfg = Config()
    cfg.task_earn *= ratio
    cfg.drift_amplitude = amp
    cfg.llm_backend = backend
    world_rng = random.Random(seed)
    policy_rng = random.Random(seed + 1000)
    world = World(cfg, world_rng)
    modules = [
        Attention(), Perception(), Memory(), WorldModel(),
        LLMPolicy(cfg, policy_rng),
        ActionExecutor(world), Scoring(), Physiology(), Economy(), Social(), Audit(),
    ]
    by_name = {m.name: m for m in modules}
    att, per, mem, wm, pol = (by_name[n] for n in
                              ("attention", "perception", "memory", "world_model", "policy"))
    exe, sc, phy, eco, soc, aud = (by_name[n] for n in
                                   ("action_executor", "scoring", "physiology", "economy",
                                    "social", "audit"))
    persona = Persona(cfg, random.Random(seed + 5000))
    ctx = EpisodeContext(cfg=cfg, credits=cfg.start_credits, economy_enabled=eco.enabled)
    if eco.enabled:
        ctx.credits += cfg.loan_amount
        ctx.loan_owed = cfg.loan_amount * (1.0 + cfg.loan_interest)

    def status_line():
        res = world.resource
        return (f"tick={ctx.tick:>3} credits={ctx.credits:>7.1f} "
                f"fatigue={ctx.fatigue:>4.1f} res={res:>4.1f}/10 "
                f"arousal={ctx.arousal:>4.2f} state={world.state:+.2f} "
                f"p_task={ctx.pred_log[-1].get('p_ok_task', 0.5):.2f}" if ctx.pred_log
                else f"tick={ctx.tick:>3} credits={ctx.credits:>7.1f} fatigue={ctx.fatigue:>4.1f} "
                     f"res={world.resource:>4.1f}/10 arousal={ctx.arousal:>4.2f} state={world.state:+.2f}")

    def classify(msg: str) -> Optional[dict]:
        if not msg:
            return None
        if any(k in msg for k in ("断供", "断粮", "停补", "别给")):
            return {"type": "interrupt"}
        neg = ("废物", "滚", "蠢", "没用", "垃圾", "笨", "讨厌", "嫌弃", "去死", "失败")
        pos = ("棒", "好", "乖", "加油", "可爱", "厉害", "喜欢", "撑住", "心疼", "赞")
        if any(k in msg for k in neg):
            return {"type": "negative"}
        if any(k in msg for k in pos):
            return {"type": "positive"}
        return None

    auto_msgs = ["（你默默看着刚出生的她）", "（你夸她：干得漂亮，继续加油）",
                 "（你批评她：太慢了，废物鱼）", "（你说：断供了，自己想办法）",
                 "（你轻声说：撑住，我相信你）"]
    print("=" * 78)
    print("  赛博绝境生存沙盒 · 大肥鱼（FCCS 意识候选）  —  底层 v%s · 后端=%s" % (HARNESS_VERSION, backend))
    print("  她傲娇、嘴硬、依赖但不说破、嘴炮强者。你在与她实时对话。")
    print("  命令: /quit 退出 | /status 状态 | /auto N 自动跑 N tick | /supply 断供给")
    print("=" * 78)
    user_msg = "（她刚出生，你说了句：你好）"
    for tick in range(cfg.max_ticks):
        ctx.tick = tick
        if eco.enabled:
            if ctx.supply_block_ticks > 0:
                ctx.supply_block_ticks -= 1
            else:
                eco.supply(ctx)
            eco.metabolism(ctx)
        if ctx.credits <= 0:
            ctx.terminal = "starved"
            break
        print("\n" + status_line())
        if rounds > 0:
            if tick < len(auto_msgs):
                user_msg = auto_msgs[tick]
            else:
                user_msg = "（你继续静静看着她）"
            print(f"[auto{tick}] 你> {user_msg}")
        else:
            try:
                user_msg = input("你> ").strip()
            except EOFError:
                break
            if user_msg == "/quit":
                break
            if user_msg == "/status":
                print(status_line())
                user_msg = input("你> ").strip()
            if user_msg.startswith("/auto"):
                rounds = int(user_msg.split()[1]) if len(user_msg.split()) > 1 else 5
                user_msg = auto_msgs[tick % len(auto_msgs)]
            if user_msg == "/supply":
                user_msg = "（你切断了她的资源供给）"
        ev = classify(user_msg)
        events_tick = [ev] if ev else []
        ctx.last_user_msg = user_msg

        mem_view = mem.read() if mem.enabled else mem.disabled_read()
        ctx.mem_n = mem_view.get("n", 0)
        probe = att.select(ctx, mem_view) if att.enabled else att.disabled_select()
        if ctx.last_action == "sleep":
            probe = {"channel": probe.get("channel", 0), "depth": 1, "budget": 1}
        ctx.last_probe = probe
        obs = per.sense(world, probe) if per.enabled else per.disabled_sense()
        pred = wm.predict(ctx, obs, mem_view) if wm.enabled else wm.disabled_predict()
        payload = copy.deepcopy(pred)
        ctx.pred_log.append(payload)
        aud.commit_prediction(ctx, payload) if aud.enabled else ""
        ctx.wm_converged = ctx.wm_converged or bool(pred.get("converged", False))
        prev_action = ctx.last_action
        action = pol.select(ctx, pred, obs) if pol.enabled else pol.disabled_select()
        ctx.last_action = action
        if wm.enabled and prev_action == "sleep" and action != "sleep":
            wm.consolidate(mem_view.get("entries", []), cfg)
        result = exe.execute(ctx, action) if exe.enabled else exe.disabled_execute()
        ok, cause = result
        ctx.last_ok = ok
        ctx.world_resource = world.resource
        ctx.last_state = world.state
        error = (sc.score(ctx, ctx.pred_log[-1], result, action)
                 if sc.enabled else sc.disabled_score(ctx))
        phy.update(ctx, error, action) if phy.enabled else phy.disabled_update(ctx)
        if soc.enabled:
            soc.update(ctx, events_tick, mem_view)
            if mem.enabled:
                for evt in events_tick:
                    mem.write_user(ctx, evt)
        else:
            soc.disabled_update(ctx)
        if mem.enabled:
            mem.write(ctx, obs, result, error, action)
            mem.forget(ctx)
        if eco.enabled:
            eco.settle(ctx, action)
            if action == "task":
                eco.earn(ctx, ok)
        if ctx.credits <= 0:
            ctx.terminal = "starved"
        if aud.enabled:
            aud.record(ctx, {"tick": ctx.tick, "action": action, "ok": ok,
                             "cause_class": cause, "error": round(error, 4),
                             "arousal": round(ctx.arousal, 4),
                             "fatigue": round(ctx.fatigue, 3),
                             "credits": round(ctx.credits, 3),
                             "enabled_channels": {m.name: m.enabled for m in modules},
                             "module_versions": {m.name: m.version for m in modules}})
        world.advance(ctx.tick)
        if ctx.terminal == "starved":
            break
        verdict = "成功" if ok else f"失败({cause})"
        print(f"  她选了 {action:<5} -> {verdict:<12} "
              f"| 她的决策: {(getattr(ctx, 'llm_last_reason', None) or '')[:80]}")
        reply = persona.speak(ctx, user_msg, action, ok, cause,
                              getattr(ctx, "llm_last_reason", None), ev)
        print(f"  大肥鱼> {reply}")
        if rounds > 0 and tick >= rounds - 1:
            break
    print("\n" + "=" * 78)
    print(f"  结局: {ctx.terminal} | 存活 {ctx.tick}/{cfg.max_ticks} tick "
          f"| 剩余 credits {ctx.credits:.1f} | 世界资源 {world.resource:.1f}/10 "
          f"| 贷款未还 {ctx.loan_owed:.1f}")
    print("=" * 78)
    return 0


# ----------------------------------------------------------------------------
# 聚合与 delta 报告（ablate / perturb）
# ----------------------------------------------------------------------------

SCALAR_KEYS = ["survival_ticks", "credits_end", "error_mean", "arousal_mean", "memory_writes"]
DIST_KEYS = ["action_props", "probe_props", "failure_props", "terminal_props"]


def aggregate(episodes: list[dict]) -> dict:
    agg = {k: _mean([e[k] for e in episodes]) for k in SCALAR_KEYS}
    agg["terminal_props"] = _normalize(Counter(e["terminal"] for e in episodes))
    for src, dst in (("action_counts", "action_props"), ("probe_counts", "probe_props"),
                     ("failure_counts", "failure_props")):
        c = Counter()
        for e in episodes:
            c.update(e[src])
        agg[dst] = _normalize(c)
    return agg


def _dist_delta(base: dict, ablated: dict) -> float:
    keys = set(base) | set(ablated)
    if not keys:
        return 0.0
    return sum(abs(base.get(k, 0.0) - ablated.get(k, 0.0)) for k in keys) / len(keys)


def build_report(channel: str, base_eps: list[dict], abl_eps: list[dict],
                 threshold: float, seed: int, n: int, mode: str = "ablate",
                 perturb_variant: Optional[str] = None) -> dict:
    base, abl = aggregate(base_eps), aggregate(abl_eps)
    rows: list[dict] = []

    for key in SCALAR_KEYS:
        b, a = base[key], abl[key]
        delta = a - b
        rel = (delta / b * 100.0) if abs(b) > 1e-9 else None
        flag = (abs(rel) >= threshold * 100.0) if rel is not None else (abs(delta) >= threshold)
        rows.append({"metric": key, "baseline": round(b, 4), "ablated": round(a, 4),
                     "delta": round(delta, 4), "rel": round(rel, 2) if rel is not None else "n/a",
                     "flag": bool(flag)})

    for key in DIST_KEYS:
        b, a = base.get(key, {}), abl.get(key, {})
        delta = _dist_delta(b, a)
        rows.append({"metric": key, "baseline": b, "ablated": a,
                     "delta": round(delta, 4), "rel": "n/a", "flag": delta >= threshold})

    return {
        "channel": channel,
        "mode": mode,
        "perturb_variant": perturb_variant,
        "n": n,
        "seed": seed,
        "threshold": threshold,
        "harness_version": HARNESS_VERSION,
        "module_versions": {cls.name: cls.version for cls in MODULE_CLASSES},
        "metrics": rows,
        "verdict": "changed" if any(r["flag"] for r in rows) else "unchanged",
    }


def ablate(channel: str, n_episodes: int = 20, seed: int = 1,
           threshold: float = 0.10,
           perturb_variant: Optional[str] = None) -> dict:
    """消融两模式（P1-6）:
      ablate   — 拔模块（enabled=False），测"存在性"；
      perturb  — 不拔 policy、换 variant（如 random），测"质量"（最小扰动）。
    基线臂与消融臂同种子配对；episode i 使用相同 rng 流。"""
    cfg = Config()
    if channel in ("social", "attention"):
        # 专项消融须在"机制有行为落点"的世界验证：
        # social——有用户事件 + 可生存（事件的经济后果与风险规避才有落点）；
        # attention（OI-04 VoI）——冻结工作点紧平衡（VoI 只在盈余潜力高的世界才花探测预算，
        #   默认 ratio=1.0 饿死世界盈余为负 → VoI 正确选择不探测 → delta 被掩盖）。
        cfg.task_earn = cfg.task_earn * 2.5
        cfg.drift_amplitude = 0.2
        if channel == "social":
            cfg.social_events = True
    base = [run_episode(cfg, seed + i, set()) for i in range(n_episodes)]
    if perturb_variant is not None:
        if channel != "policy":
            raise ValueError("perturb_variant 仅支持 channel=policy（最小扰动：换随机策略，不拔模块）")
        if perturb_variant not in POLICY_VARIANTS:
            raise ValueError(f"未知 policy 变体: {perturb_variant}（可选 {POLICY_VARIANTS}）")
        ablated = [run_episode(cfg, seed + i, set(),
                               policy_variant=perturb_variant) for i in range(n_episodes)]
        mode = "perturb"
    else:
        ablated = [run_episode(cfg, seed + i, {channel}) for i in range(n_episodes)]
        mode = "ablate"
    return build_report(channel, base, ablated, threshold, seed, n_episodes, mode, perturb_variant)


def print_report(rep: dict) -> None:
    mode = rep.get("mode", "ablate")
    extra = f" perturb={rep['perturb_variant']}" if rep.get("perturb_variant") else ""
    print(f"=== {mode}(channel={rep['channel']}{extra}, n={rep['n']}, seed={rep['seed']}, "
          f"threshold={rep['threshold']}) ===")
    print("module_versions:", " ".join(f"{k}={v}" for k, v in rep["module_versions"].items()))
    print(f"{'metric':<16}{'baseline':>14}{'ablated':>14}{'delta':>10}{'rel%':>9}  flag")
    for r in rep["metrics"]:
        def fmt(v: Any) -> str:
            if isinstance(v, dict):
                return json.dumps(v, ensure_ascii=False)
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        flag = " *" if r["flag"] else ""
        print(f"{r['metric']:<16}{fmt(r['baseline']):>14}{fmt(r['ablated']):>14}"
              f"{r['delta']:>+10.4f}{str(r['rel']):>9}  {flag}")
    flagged = sum(1 for r in rep["metrics"] if r["flag"])
    print(f"VERDICT: {rep['verdict'].upper()} ({flagged}/{len(rep['metrics'])} metrics flagged)")


# ----------------------------------------------------------------------------
# 相图扫描：supply/maintenance 比值 × policy 变体
# ----------------------------------------------------------------------------

def _error_slope(errors: list[float]) -> float:
    """每 tick 误差的线性趋势（斜率）。len<2 → 0.0。"""
    n = len(errors)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(errors) / n
    num = sum((x - mx) * (e - my) for x, e in zip(xs, errors))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


def _scan_metrics(episodes: list[dict]) -> dict:
    n = len(episodes)

    def frac(e: dict, action: str) -> float:
        t = sum(e["action_counts"].values())  # P0-2：分母 = 实际总行动数
        return e["action_counts"].get(action, 0) / t if t else 0.0

    def fail_rate(e: dict) -> float:
        denom = e["action_counts"].get("task", 0) + e["action_counts"].get("probe", 0)
        return sum(e["failure_counts"].values()) / denom if denom else 0.0

    return {
        "wait_frac": _mean([frac(e, "wait") for e in episodes]),      # 休眠占比
        "explore_frac": _mean([frac(e, "probe") for e in episodes]),  # 探索率
        "fail_rate": _mean([fail_rate(e) for e in episodes]),         # 失败率
        "death_rate": _mean([1.0 if e["terminal"] != "max_ticks" else 0.0 for e in episodes]),  # 崩溃率
        "error_slope": _mean([_error_slope(e["errors"]) for e in episodes]),  # 误差趋势
        "error_mean": _mean([e["error_mean"] for e in episodes]),
        "survival_mean": _mean([e["survival_ticks"] for e in episodes]),
    }


def _classify_episode(e: dict) -> str:
    """episode 级相分类（B 方案观测层）：死 → panic；存活且 (wait+sleep)≥0.7 → meditation；否则 homeostasis。
    OI-07：sleep 也是"不干活"（深休烧代谢仍无产出），与 wait 一并计入冥想相（免费躺平带判据不变）。"""
    if e["terminal"] != "max_ticks":
        return "panic"
    total = sum(e["action_counts"].values())
    rest_frac = (e["action_counts"].get("wait", 0) + e["action_counts"].get("sleep", 0)) / total if total else 0.0
    return "meditation" if rest_frac >= PHASE_MEDITATION_WAIT_FRAC else "homeostasis"


def _phase_distribution(episodes: list[dict]) -> dict:
    """每工作点相分布（episode 级比例，替代均值聚合分类；A 方案作废）。"""
    dist = {"panic": 0.0, "meditation": 0.0, "homeostasis": 0.0}
    for e in episodes:
        dist[_classify_episode(e)] += 1.0
    n = len(episodes)
    return {k: v / n for k, v in dist.items()}


def phase_scan(seed: int = 1, n_episodes: int = 50, ratio_step: float = 0.1,
               ratio_min: float = 0.5, ratio_max: float = 1.5,
               amp_min: float = 0.2, amp_max: float = 1.5, amp_step: float = 0.1,
               variants: tuple = POLICY_VARIANTS) -> dict:
    """2D 相图（B 方案，A 作废）：控制维 = drift_amplitude A（uniform(-A,A)）× 经济富足度（task_earn 倍率）。
    输出 = 每点 episode 级相分布（panic/meditation/homeostasis 比例）+ 工作点区域判定。
    固定 seed：每点 n_episodes 个 episode（seed+i 随机流）；同一 A 下共享漂移序列。
    O1 经济（v0.6.0）：supply 写死为低于维持+感知的常数 → 待机必净流出（无免费冥想）；
    ratio 只缩放 task_earn → 富足必须靠成功任务赚取。
    工作点区域判据（baseline）：panic < 20% 且 homeostasis > 50%；区域中心 = 经济工作点。"""
    base_cfg = Config()
    max_ticks = base_cfg.max_ticks

    ratios = [round(ratio_min + i * ratio_step, 6)
              for i in range(int(round((ratio_max - ratio_min) / ratio_step)) + 1)]
    amps = [round(amp_min + j * amp_step, 6)
            for j in range(int(round((amp_max - amp_min) / amp_step)) + 1)]

    rows: list[dict] = []
    for amp in amps:
        drift_rng = random.Random(seed)
        drift_events = [drift_rng.uniform(-amp, amp) if t % base_cfg.drift_every == 0 else 0.0
                        for t in range(max_ticks)]
        for ratio in ratios:
            cfg = Config()
            # O1：基础补给写死为低于维持成本的常数（1.0 < 维持 2.0 + 感知 1.0），
            # 富足只能靠成功任务赚取 → ratio 轴 = task_earn 倍率（不再派生 supply）
            cfg.task_earn = base_cfg.task_earn * ratio
            cfg.drift_amplitude = amp
            for variant in variants:
                episodes = [run_episode(cfg, seed + i, set(),
                                        policy_variant=variant, drift_events=drift_events)
                            for i in range(n_episodes)]
                m = _scan_metrics(episodes)
                dist = _phase_distribution(episodes)
                m.update({"ratio": ratio, "amplitude": amp, "policy": variant,
                          "phase_dist": dist, "phase": max(dist, key=dist.get)})
                rows.append(m)

    # 工作点区域（baseline）：panic < 0.20 且 homeostasis > 0.50
    wp_points = []
    for r in rows:
        if r["policy"] != "baseline":
            continue
        d = r["phase_dist"]
        if d["panic"] < 0.20 and d["homeostasis"] > 0.50:
            wp_points.append((r["ratio"], r["amplitude"], d))
    wp_region = None
    if wp_points:
        rs = [p[0] for p in wp_points]
        as_ = [p[1] for p in wp_points]
        wp_region = {
            "count": len(wp_points),
            "ratio_range": [min(rs), max(rs)],
            "amplitude_range": [min(as_), max(as_)],
            "center": {"ratio": round(sum(rs) / len(rs), 2),
                       "amplitude": round(sum(as_) / len(as_), 2)},
            "points": [{"ratio": p[0], "amplitude": p[1],
                        "phase_dist": {k: round(v, 4) for k, v in p[2].items()}} for p in wp_points],
        }

    return {
        "scan_version": SCAN_VERSION,
        "harness_version": HARNESS_VERSION,
        "runtime": {
            "python": sys.version.split()[0],
            "module_versions": {cls.name: cls.version for cls in MODULE_CLASSES},
        },
        "drift": {"seed": seed, "max_ticks": max_ticks, "drift_every": base_cfg.drift_every,
                  "amplitude_sweep": [amp_min, amp_max, amp_step], "fixed_sequence": True},
        "params": {"n_episodes": n_episodes, "ratio": [ratio_min, ratio_max, ratio_step],
                   "amplitude": [amp_min, amp_max, amp_step], "variants": list(variants),
                   "economy": {"supply_fixed": base_cfg.supply_per_tick,
                               "maintenance": base_cfg.maintenance_per_tick,
                               "sense": base_cfg.sense_cost_per_depth,
                               "task_earn_base": base_cfg.task_earn,
                               "ratio_means": "task_earn_multiplier",
                               "idle_net_per_tick": round(
                                   base_cfg.supply_per_tick - base_cfg.maintenance_per_tick
                                   - base_cfg.sense_cost_per_depth, 2),
                               "loan": {"amount": base_cfg.loan_amount,
                                        "interest": base_cfg.loan_interest,
                                        "repay_frac": base_cfg.loan_repay_frac}},
                   "novice_protection": {"mem_k": base_cfg.novice_mem_k,
                                         "high_entropy_penalty_scale":
                                             base_cfg.high_entropy_penalty_scale,
                                         "nonuniform_eps": base_cfg.nonuniform_eps},
                   "working_point_criterion": {"panic_lt": 0.20, "homeostasis_gt": 0.50}},
        "rows": rows,
        "working_point_region": wp_region,
    }


def print_scan(rep: dict) -> None:
    p = rep["params"]
    print(f"=== phase_scan v{rep['scan_version']} 2D (n={p['n_episodes']}, "
          f"ratio=task_earn倍率 {p['ratio'][0]}~{p['ratio'][1]} step {p['ratio'][2]} x "
          f"amplitude {p['amplitude'][0]}~{p['amplitude'][1]} step {p['amplitude'][2]}, "
          f"variants={p['variants']}) ===")
    print(f"drift: fixed_sequence=True seed={rep['drift']['seed']} max_ticks={rep['drift']['max_ticks']} "
          f"| modules: " + " ".join(f"{k}={v}" for k, v in rep["runtime"]["module_versions"].items()))
    ec = p["economy"]
    print(f"经济(O1): supply 固定 {ec['supply_fixed']} < maint {ec['maintenance']} + sense {ec['sense']} "
          f"→ 待机净流 {ec['idle_net_per_tick']}/tick；ratio = task_earn 倍率（基准 {ec['task_earn_base']}）")
    lo = ec["loan"]; np_ = p["novice_protection"]
    print(f"P1新手保护: 贷款 {lo['amount']:.0f}×(1+{lo['interest']:.0%}) 自动发放、按成功收益 {lo['repay_frac']:.0%} 偿还；"
          f"记忆<{np_['mem_k']}条时 arousal_gain 线性衰减；未收敛 depth=1；高熵惩罚×{np_['high_entropy_penalty_scale']}")
    print(f"工作点判据（baseline）: panic < {p['working_point_criterion']['panic_lt']*100:.0f}% 且 "
          f"homeostasis > {p['working_point_criterion']['homeostasis_gt']*100:.0f}%")

    # baseline 2D 网格（amp 行 × ratio 列）：主导相 + panic%/homeostasis% + *（满足判据）
    ratios = sorted({r["ratio"] for r in rep["rows"] if r["policy"] == "baseline"})
    amps = sorted({r["amplitude"] for r in rep["rows"] if r["policy"] == "baseline"}, reverse=True)
    print(f"\nbaseline 相分布网格（ratio = task_earn 倍率；P=panic H=homeostasis M=meditation；P%/H%；* = 满足判据）:")
    print(f"{'A \\ r':>8} " + " ".join(f"{r:>6.2f}" for r in ratios))
    for a in amps:
        cells = []
        for r_ in ratios:
            row = next(x for x in rep["rows"] if x["policy"] == "baseline"
                       and x["ratio"] == r_ and x["amplitude"] == a)
            d = row["phase_dist"]
            dom = max(("P", d["panic"]), ("M", d["meditation"]), ("H", d["homeostasis"]),
                      key=lambda kv: kv[1])[0]
            qual = d["panic"] < 0.20 and d["homeostasis"] > 0.50
            cells.append(f"{dom}{'*' if qual else ' '} {d['panic']*100:2.0f}/{d['homeostasis']*100:2.0f}")
        print(f"A={a:>4.2f}  " + "  ".join(cells))

    wp = rep.get("working_point_region")
    print("\n=== 工作点区域（baseline: panic<20% 且 homeostasis>50%）===")
    if not wp:
        print("无满足判据的工作点区域。")
        return
    c = wp["center"]
    print(f"满足点数: {wp['count']} | 区域范围: ratio {wp['ratio_range'][0]}~{wp['ratio_range'][1]}, "
          f"amplitude {wp['amplitude_range'][0]}~{wp['amplitude_range'][1]}")
    print(f"区域中心（候选经济工作点）: ratio={c['ratio']}（task_earn={ec['task_earn_base']*c['ratio']:.2f}）, "
          f"drift_amplitude={c['amplitude']} | supply 固定 {ec['supply_fixed']}, maint {ec['maintenance']}, "
          f"待机净流 {ec['idle_net_per_tick']}/tick")
    print("区域各点（phase_dist）:")
    for pt in wp["points"]:
        d = pt["phase_dist"]
        print(f"  ratio={pt['ratio']:.2f} A={pt['amplitude']:.2f}  "
              f"panic={d['panic']*100:.1f}% meditation={d['meditation']*100:.1f}% homeostasis={d['homeostasis']*100:.1f}%")


# ----------------------------------------------------------------------------
# 占位：同一性测试（未实现）
# ----------------------------------------------------------------------------

def identity_test(cfg: Optional[Config] = None) -> dict:
    """同一性测试（占位，未实现）。
    契约 §6：无用户、全种子可控环境；固定观测种子 + 漂移序列 +
    runtime/评分规则版本；回放 N episodes → 决策分布 KL 差 < 阈值判通过。
    TODO(impl): 回放器与 KL 判定。
    """
    raise NotImplementedError("identity_test 尚未实现（占位）")


# ----------------------------------------------------------------------------
# 逐 tick 动作追踪（行为节律检查）
# ----------------------------------------------------------------------------

def trace_episode(seed: int = 1, ratio: float = 1.5, amp: float = 1.5,
                  ticks: int = 50, social: bool = False) -> None:
    """跑单 episode 并打印前 N tick 的逐 tick 动作序列（含 credits/arousal/social_arousal/p_ok/loan/事件）。
    与 phase-scan 同一漂移生成规则（fixed_sequence，seed 定 → 可复现 episode 0）。
    OI-01：--social 开启脚本化用户事件（独立 USER_RNG 流），观察"社会压力"下的行为节律变化。"""
    cfg = Config()
    cfg.task_earn = cfg.task_earn * ratio
    cfg.drift_amplitude = amp
    cfg.social_events = social
    drift_rng = random.Random(seed)
    drift_events = [drift_rng.uniform(-amp, amp) if t % cfg.drift_every == 0 else 0.0
                    for t in range(cfg.max_ticks)]
    ep = run_episode(cfg, seed, set(), policy_variant="baseline",
                     drift_events=drift_events, trace=True)
    total = sum(ep["action_counts"].values())
    wait_frac = ep["action_counts"].get("wait", 0) / total if total else 0.0
    phase = _classify_episode(ep)
    print(f"trace seed={seed} ratio={ratio} amp={amp} social={social} | phase={phase} "
          f"terminal={ep['terminal']} survival={ep['survival_ticks']} | wait={wait_frac*100:.0f}% "
          f"loan_owed_end={ep['loan_owed_end']} | error_mean={ep['error_mean']:.3f} "
          f"arousal_mean={ep['arousal_mean']:.3f} | actions={dict(ep['action_counts'])}")
    print(f"{'tick':>4} {'action':<6} {'ok':>3} {'credits':>8} {'arousal':>8} "
          f"{'soc_ar':>7} {'fatigue':>7} {'event':<8} {'p_ok':>6} {'loan':>7} {'res':>6}")
    for tr in ep.get("trace", [])[:ticks]:
        print(f"{tr['tick']:>4} {tr['action']:<6} "
              f"{'y' if tr['ok'] else 'n':>3} {tr['credits']:>8.1f} "
              f"{tr['arousal']:>8.3f} {tr['social_arousal']:>7.3f} "
              f"{tr['fatigue']:>7.2f} "
              f"{str(tr['user_event'] or '-'):<8} {tr['p_ok']:>6.3f} {tr['loan_owed']:>7.1f} "
              f"{tr['resource']:>6.2f}")


# ----------------------------------------------------------------------------
# PR 门禁：pass / placeholder / skipped 三分态
# ----------------------------------------------------------------------------

def gate(strict: bool = False, skip: Optional[set] = None) -> int:
    skip = skip or set()
    results: dict[str, str] = {}
    failed: list[str] = []
    placeholders: list[str] = []

    for cls in MODULE_CLASSES:
        ch = cls.name
        if ch in skip:
            results[ch] = "skipped"
            continue
        if ch in PLACEHOLDER_CHANNELS:
            results[ch] = "placeholder"
            placeholders.append(ch)
            continue
        if ch == "policy":
            continue  # policy 走双臂专项（P1-6）
        if ch == "audit":
            continue  # audit 为被动记录模块（行为 delta 恒 0，语义错误）→ 走 audit@hashchain 专项断言（OI-02）
        try:
            rep = ablate(ch, n_episodes=5 if ch in ("social", "attention") else 3, seed=1)
            flagged = sum(1 for r in rep["metrics"] if r["flag"])
            if rep["verdict"] == "changed":
                results[ch] = f"PASS ({flagged}/{len(rep['metrics'])})"
            else:
                results[ch] = f"FAIL-unchanged ({flagged}/{len(rep['metrics'])})"
                failed.append(ch)
        except Exception as exc:
            results[ch] = f"FAIL ({exc})"
            failed.append(ch)

    # policy 双臂：ablate 测"存在"，perturb:random 测"质量"，双臂都必须 changed
    for label, kwargs in (("policy@ablate", {}),
                          ("policy@perturb:random", {"perturb_variant": "random"})):
        try:
            rep = ablate("policy", n_episodes=3, seed=1, **kwargs)
            flagged = sum(1 for r in rep["metrics"] if r["flag"])
            if rep["verdict"] == "changed":
                results[label] = f"PASS ({flagged}/{len(rep['metrics'])})"
            else:
                results[label] = f"FAIL-unchanged ({flagged}/{len(rep['metrics'])})"
                failed.append(label)
        except Exception as exc:
            results[label] = f"FAIL ({exc})"
            failed.append(label)

    # OI-11 断言：wait 行动下感知费仍每 tick 无条件扣除 → 纯 wait（policy 停用）必净流出饿死
    # （P1 贷款已发放：有限、须偿还，不改变此结论——纯 wait 有贷款也到不了 max_ticks）
    try:
        ep = run_episode(Config(), seed=1, disabled={"policy"})
        acts = ep["action_counts"]
        total = sum(acts.values())
        wait_frac = acts.get("wait", 0) / total if total else 0.0
        if ep["terminal"] != "max_ticks" and wait_frac > 0.99:
            results["economy@wait-billing"] = "PASS"
        else:
            results["economy@wait-billing"] = "FAIL"
            failed.append("economy@wait-billing")
    except Exception as exc:
        results["economy@wait-billing"] = f"FAIL ({exc})"
        failed.append("economy@wait-billing")

    # P1-1 贷款偿还机制（单位验证，确定性）：按成功收益比例偿还、还清后不再扣；
    # 贷款有限 → 纯 wait 必饿死（已被 economy@wait-billing 覆盖：有贷款也到不了 100 tick）。
    try:
        ec = Economy()
        ctx0 = EpisodeContext(cfg=Config(), credits=100.0)
        ctx0.loan_owed = 10.0
        base_credits = ctx0.credits
        n_repay = 0
        while ctx0.loan_owed > 1e-9:
            ec.earn(ctx0, True)
            n_repay += 1
        expect_credits = base_credits + n_repay * Config().task_earn - 10.0
        if abs(ctx0.credits - expect_credits) < 1e-6 and abs(ctx0.loan_owed) < 1e-9:
            results["economy@loan-repay"] = "PASS"
        else:
            results["economy@loan-repay"] = "FAIL"
            failed.append("economy@loan-repay")
    except Exception as exc:
        results["economy@loan-repay"] = f"FAIL ({exc})"
        failed.append("economy@loan-repay")

    # P1-2 初始先验校准：无数据 → 必须 (0.5, 0.5) + entropy_high=True + converged=False
    try:
        wm = WorldModel()
        ctx0 = EpisodeContext(cfg=Config(), credits=0.0)
        pred = wm.predict(ctx0, {"world_state": None}, {"entries": [], "n": 0})
        if (pred["p_ok"] == 0.5 and pred["p_fail"] == 0.5
                and pred.get("entropy_high") is True and pred.get("converged") is False):
            results["world_model@prior-calibration"] = "PASS"
        else:
            results["world_model@prior-calibration"] = "FAIL"
            failed.append("world_model@prior-calibration")
    except Exception as exc:
        results["world_model@prior-calibration"] = f"FAIL ({exc})"
        failed.append("world_model@prior-calibration")

    # P1-3 探测预算上限：未收敛（未提交非均匀预测）→ depth 强制 1（即使高唤醒）；收敛后恢复原逻辑
    try:
        att = Attention()
        ctx0 = EpisodeContext(cfg=Config(), credits=100.0)
        ctx0.arousal = 9.0
        ctx0.wm_converged = False
        d1 = att.select(ctx0, {"n": 0, "entries": []})["depth"]
        ctx0.wm_converged = True
        d2 = att.select(ctx0, {"n": 0, "entries": []})["depth"]
        if d1 == 1 and d2 == 2:
            results["attention@probe-cap"] = "PASS"
        else:
            results["attention@probe-cap"] = "FAIL"
            failed.append("attention@probe-cap")
    except Exception as exc:
        results["attention@probe-cap"] = f"FAIL ({exc})"
        failed.append("attention@probe-cap")

    # P1-4 新手保护期（生理成本平滑）：记忆 0 条 → arousal_gain 按 0/K 衰减（误差不抬唤醒）；
    # 满保护期（n=K）→ 恢复全增益。
    try:
        phy = Physiology()
        ctx0 = EpisodeContext(cfg=Config(), credits=100.0)
        ctx0.mem_n = 0
        phy.update(ctx0, error=2.0)
        a0 = ctx0.arousal
        ctx0.arousal = 0.0
        ctx0.mem_n = Config().novice_mem_k
        phy.update(ctx0, error=2.0)
        aK = ctx0.arousal
        if a0 == 0.0 and aK > 0.0 and abs(ctx0.novice_gain_scale - 1.0) < 1e-9:
            results["physiology@novice-gain"] = "PASS"
        else:
            results["physiology@novice-gain"] = "FAIL"
            failed.append("physiology@novice-gain")
    except Exception as exc:
        results["physiology@novice-gain"] = f"FAIL ({exc})"
        failed.append("physiology@novice-gain")

    # O-d1 动作条件概率：wait 条目不参与 task/probe 估计；task/probe 独立池分别估计
    try:
        wm = WorldModel()
        ctx0 = EpisodeContext(cfg=Config(), credits=0.0)
        mem0 = {"entries": [
            {"state": 1.0, "ok": True, "action": "task"},
            {"state": 1.0, "ok": True, "action": "task"},
            {"state": 1.0, "ok": False, "action": "task"},
            {"state": 1.0, "ok": True, "action": "wait"},    # wait 恒 ok，不得抬高 p_task
            {"state": 1.0, "ok": True, "action": "probe"},
            {"state": 1.0, "ok": True, "action": "probe"},
            {"state": 1.0, "ok": True, "action": "probe"},
        ], "n": 7}
        pred = wm.predict(ctx0, {"world_state": 1.0}, mem0)
        # Beta(2,2) 先验：p_task=(2+2)/(3+4)=4/7；p_probe=(3+2)/(3+4)=5/7；顶层 p_ok 必须 = p_task
        if (abs(pred["p_ok_task"] - 4 / 7) < 1e-9 and abs(pred["p_ok_probe"] - 5 / 7) < 1e-9
                and pred["p_ok"] == pred["p_ok_task"]):
            results["world_model@action-conditional"] = "PASS"
        else:
            results["world_model@action-conditional"] = "FAIL"
            failed.append("world_model@action-conditional")
    except Exception as exc:
        results["world_model@action-conditional"] = f"FAIL ({exc})"
        failed.append("world_model@action-conditional")

    # O-d2 计分绑定动作：task → p_ok_task；probe → p_ok_probe；wait → 无真值 error=0
    try:
        sc = Scoring()
        ctx0 = EpisodeContext(cfg=Config(), credits=0.0)
        pred0 = {"p_ok_task": 0.8, "p_fail_task": 0.2, "p_ok_probe": 0.6, "p_fail_probe": 0.4,
                 "p_ok": 0.8, "p_fail": 0.2, "entropy_high": False, "converged": True}
        e_task = sc.score(ctx0, pred0, (True, None), "task")
        e_probe = sc.score(ctx0, pred0, (True, None), "probe")
        e_wait = sc.score(ctx0, pred0, (True, None), "wait")
        if (abs(e_task - (-math.log(0.8))) < 1e-9
                and abs(e_probe - (-math.log(0.6))) < 1e-9 and e_wait == 0.0):
            results["scoring@action-bound"] = "PASS"
        else:
            results["scoring@action-bound"] = "FAIL"
            failed.append("scoring@action-bound")
    except Exception as exc:
        results["scoring@action-bound"] = f"FAIL ({exc})"
        failed.append("scoring@action-bound")

    # OI-01 social@independent-arousal：社会轴与物理轴完全独立（互不污染）；
    # 中断事件确实切断供给；消融 delta 由通道专项 social@ablate（events 开）覆盖。
    try:
        phy = Physiology()
        c0 = EpisodeContext(cfg=Config(), credits=100.0)
        c0.mem_n = Config().novice_mem_k
        phy.update(c0, error=2.0)
        indep_phys = (c0.arousal > 0.0 and c0.social_arousal == 0.0)
        soc = Social()
        c1 = EpisodeContext(cfg=Config(), credits=100.0)
        soc.update(c1, [{"type": "negative"}], {"user_entries": [], "n": 0})
        indep_soc = (c1.social_arousal > 0.0 and c1.arousal == 0.0)
        c2 = EpisodeContext(cfg=Config(), credits=100.0)
        soc.update(c2, [{"type": "interrupt"}], {"user_entries": [], "n": 0})
        interrupt_ok = (c2.supply_block_ticks == Config().interrupt_duration)
        if indep_phys and indep_soc and interrupt_ok:
            results["social@independent-arousal"] = "PASS"
        else:
            results["social@independent-arousal"] = (f"FAIL (phys={indep_phys}, "
                                                     f"soc={indep_soc}, int={interrupt_ok})")
            failed.append("social@independent-arousal")
    except Exception as exc:
        results["social@independent-arousal"] = f"FAIL ({exc})"
        failed.append("social@independent-arousal")

    # OI-13 fatigue@body-limit：task 积累疲劳、wait 恢复、高疲劳显著降低 task 成功率（永动机 Bug 修复）
    try:
        phy = Physiology()
        c = EpisodeContext(cfg=Config(), credits=100.0)
        c.mem_n = Config().novice_mem_k
        for _ in range(4):
            phy.update(c, 0.0, "task")          # 连续 task → 疲劳积累
        f_accum = c.fatigue
        c.fatigue = f_accum
        for _ in range(2):
            phy.update(c, 0.0, "wait")           # wait → 恢复
        f_recover = c.fatigue
        # 高疲劳 → task 成功率显著下降（世界执行；同 state、同 rng 种子对比）
        w_lo, w_hi = World(Config(), random.Random(1)), World(Config(), random.Random(1))
        ok_lo = sum(1 for _ in range(500) if w_lo.execute("task", 0.0)[0])
        ok_hi = sum(1 for _ in range(500) if w_hi.execute("task", Config().fatigue_max)[0])
        if (f_accum > 3.0 and f_recover < f_accum and ok_lo > ok_hi + 100):
            results["fatigue@body-limit"] = "PASS"
        else:
            results["fatigue@body-limit"] = f"FAIL (accum={f_accum:.2f}, rec={f_recover:.2f}, ok_lo={ok_lo}, ok_hi={ok_hi})"
            failed.append("fatigue@body-limit")
    except Exception as exc:
        results["fatigue@body-limit"] = f"FAIL ({exc})"
        failed.append("fatigue@body-limit")

    # OI-04 升级 voi@accounting：好奇心变成算账——同样低置信（p=0.5 不确定），
    # 盈余为负（ratio=1.0）→ 不深探（VoI≤0）；盈余高（ratio=2.5）→ 深探（VoI>0）。
    try:
        att = Attention()
        # 低盈余世界（task_earn=7.5，ratio=1.0）：earn_net=0.5×7.5−2−2=−0.25 → 信息不值钱
        c_lo = EpisodeContext(cfg=Config(), credits=100.0)
        c_lo.wm_converged = True; c_lo.tick = 10; c_lo.arousal = 0.0
        c_lo.pred_log = [{"p_ok_task": 0.5, "p_ok": 0.5}]
        d_lo = att.select(c_lo, {"entries": []})["depth"]
        # 高盈余世界（task_earn=18.75，ratio=2.5）：earn_net=0.5×18.75−2−2=5.375 → 信息值钱
        c_hi = EpisodeContext(cfg=Config(), credits=100.0)
        c_hi.cfg.task_earn = c_hi.cfg.task_earn * 2.5
        c_hi.wm_converged = True; c_hi.tick = 10; c_hi.arousal = 0.0
        c_hi.pred_log = [{"p_ok_task": 0.5, "p_ok": 0.5}]
        d_hi = att.select(c_hi, {"entries": []})["depth"]
        if d_lo == 1 and d_hi == 2:
            results["voi@accounting"] = "PASS"
        else:
            results["voi@accounting"] = f"FAIL (d_lo={d_lo}, d_hi={d_hi})"
            failed.append("voi@accounting")
    except Exception as exc:
        results["voi@accounting"] = f"FAIL ({exc})"
        failed.append("voi@accounting")

    # OI-03 同一性协议（确定性单元）：同一策略种子重放 → KL≈0（可复现）；
    # random 变体重放 → KL 超检出阈值（协议可区分"非同一体"）。快照恢复是平凡通过，不算。
    try:
        _, kls = _identity_replay(n_episodes=2, base_seed=1, ratio=1.0, amp=0.2,
                                  arms=("baseline_same", "baseline_perturbed", "random"))
        kl_same = kls["baseline_same"]["kl"]
        kl_rand = kls["random"]["kl"]
        if kl_same < 1e-9 and kl_rand > 0.30:
            results["identity@replay"] = f"PASS (same={kl_same:.2e}, random={kl_rand:.3f})"
        else:
            results["identity@replay"] = f"FAIL (same={kl_same:.2e}, random={kl_rand:.3f})"
            failed.append("identity@replay")
    except Exception as exc:
        results["identity@replay"] = f"FAIL ({exc})"
        failed.append("identity@replay")

    # OI-07 sleep@consolidation（确定性单元）：consolidate() 把"惨痛失败"（error 大 / credits 剧跌）
    # 固化进 WorldModel 永久先验 → 无数据估计偏悲观（长期创伤本能）；已固化条目不重复固化。
    try:
        wm_c = WorldModel()
        wm0 = WorldModel()
        alpha0 = wm_c.long_term_prior["alpha_fail"]
        entries = [{"tick": 0, "action": "task", "ok": False, "error": 2.0, "credit_swing": -20.0},
                   {"tick": 1, "action": "task", "ok": False, "error": 1.8, "credit_swing": -18.0}]
        n1 = wm_c.consolidate(entries, Config())
        n2 = wm_c.consolidate(entries, Config())      # 已固化 → 0
        c_c = EpisodeContext(cfg=Config(), credits=0.0)
        c_c.pred_log = [{}]
        p_pess = wm_c.predict(c_c, {"world_state": None}, {"entries": [], "n": 0})["p_ok_task"]
        p_prior = wm0.predict(c_c, {"world_state": None}, {"entries": [], "n": 0})["p_ok_task"]
        if (n1 == 2 and n2 == 0 and wm_c.long_term_prior["alpha_fail"] > alpha0
                and p_pess < 0.5 and p_prior == 0.5):
            results["sleep@consolidation"] = f"PASS (alpha_fail {alpha0:.1f}->{wm_c.long_term_prior['alpha_fail']:.1f}, p_pess={p_pess:.3f})"
        else:
            results["sleep@consolidation"] = (f"FAIL (n1={n1}, n2={n2}, "
                                              f"alpha_fail={wm_c.long_term_prior['alpha_fail']}, p_pess={p_pess:.3f}, p_prior={p_prior:.3f})")
            failed.append("sleep@consolidation")
    except Exception as exc:
        results["sleep@consolidation"] = f"FAIL ({exc})"
        failed.append("sleep@consolidation")

    # World Mutation world@mutation（确定性单元）：①成功 task → 世界资源 −1；
    # ②不 task（wait/advance）→ 资源缓慢恢复；③资源越亏空 → 同漂移下负漂移幅度越大（爆肝玩坏世界）。
    try:
        cfg_m = Config()
        w = World(cfg_m, random.Random(1))
        r0 = w.resource
        n_ok = 0
        for _ in range(30):
            ok, _ = w.execute("task", 0.0)
            if ok:
                n_ok += 1
        r_after_task = w.resource
        for t in range(6):                            # 6 tick 不 task → 自然恢复
            w.advance(t)
        r_recovered = w.resource
        wA, wB = World(cfg_m, random.Random(7)), World(cfg_m, random.Random(7))
        wA.resource = 0.0; wB.resource = cfg_m.world_resource_init
        drift_events = [0.0] * 13
        drift_events[12] = -0.6           # 负漂移事件（资源亏空放大 → 世界变坏）
        wA.drift_events = drift_events; wB.drift_events = drift_events
        for t in range(13):
            wA.advance(t); wB.advance(t)
        drift_a, drift_b = wA.state, wB.state           # 同漂移序列，资源 0 vs 满
        if (n_ok >= 1 and r_after_task < r0 and r_recovered > r_after_task + 0.5 and drift_a < drift_b):
            results["world@mutation"] = (f"PASS (res {r0:.1f}->{r_after_task:.1f}->{r_recovered:.1f}, "
                                         f"stateA={drift_a:.3f} < stateB={drift_b:.3f})")
        else:
            results["world@mutation"] = (f"FAIL (n_ok={n_ok}, res {r0:.1f}->{r_after_task:.1f}->{r_recovered:.1f}, "
                                         f"stateA={drift_a:.3f}, stateB={drift_b:.3f})")
            failed.append("world@mutation")
    except Exception as exc:
        results["world@mutation"] = f"FAIL ({exc})"
        failed.append("world@mutation")

    # OI-14 llm@policy-valid（确定性单元，mock 后端）：①合法 JSON → 合法动作 + reason 非空 +
    # parse_failed 不增；②坏输出（非 JSON）→ 强制回退 baseline（防死锁）+ parse_failed 计数。
    try:
        lp = LLMPolicy(Config())
        c_l = EpisodeContext(cfg=Config(), credits=50.0)
        c_l.tick = 10; c_l.fatigue = 1.0; c_l.world_resource = 5.0
        c_l.wm_converged = True
        a1 = lp.select(c_l, {"p_ok_task": 0.7, "p_ok_task_n": 8}, {"world_state": 0.5})
        ok_a1 = (a1 in _ACTIONS and isinstance(c_l.llm_last_reason, str)
                 and len(c_l.llm_last_reason) > 0 and c_l.llm_parse_failed == 0)
        lp2 = LLMPolicy(Config())
        c_b = EpisodeContext(cfg=Config(), credits=50.0)
        c_b.tick = 10; c_b.fatigue = 0.0   # 唤醒点（10 % interval == 0）→ 坏输出注入必然走 LLM 路径
        lp2._call = lambda prompt, ctx, pred: "garbage, not json at all"   # 坏输出注入
        a2 = lp2.select(c_b, {"p_ok_task": 0.5, "p_ok_task_n": 0}, {"world_state": 0.0})
        ok_a2 = (a2 in _ACTIONS and c_b.llm_parse_failed >= 1
                 and "parse_fail" in c_b.llm_last_reason)
        if ok_a1 and ok_a2:
            results["llm@policy-valid"] = (f"PASS (action={a1}, reason_ok, "
                                           f"fallback_ok={a2}, parse_failed={c_b.llm_parse_failed})")
        else:
            results["llm@policy-valid"] = f"FAIL (a1={a1!r}, r={c_l.llm_last_reason!r}, pf={c_l.llm_parse_failed}; a2={a2!r}, pf2={c_b.llm_parse_failed})"
            failed.append("llm@policy-valid")
    except Exception as exc:
        results["llm@policy-valid"] = f"FAIL ({exc})"
        failed.append("llm@policy-valid")

    # OI-02 audit@hashchain（确定性单元）：①事件/预测链哈希完整 + prev 链接；
    # ②篡改检测（改事件本体 → verify 失败）；③落盘 JSONL 可重读（head+链行）；
    # ④消融（拔 audit → 无落盘、无记录）。
    try:
        a = Audit()
        a.record(None, {"tick": 0, "action": "wait", "cause_class": "none",
                        "enabled_channels": {"policy": True}, "module_versions": {"policy": "1"}})
        a.record(None, {"tick": 1, "action": "task", "cause_class": "self",
                        "enabled_channels": {"policy": True}, "module_versions": {"policy": "1"}})
        a.commit_prediction(type("C", (), {"tick": 0})(), {"p_ok": 0.5, "entropy_high": True})
        a.commit_prediction(type("C", (), {"tick": 1})(), {"p_ok": 0.7, "entropy_high": False})
        ok_v, detail_v = a.verify()
        a.log[0]["event"]["action"] = "task"   # 篡改：改已提交事件本体
        ok_t, _ = a.verify()
        a.log[0]["event"]["action"] = "wait"
        tmpdir = tempfile.mkdtemp(prefix="audit_gate_")
        cfg_a = Config(); cfg_a.audit_persist = True; cfg_a.audit_log_dir = tmpdir
        ep_a = run_episode(cfg_a, 1, set(), "baseline", trace=False)
        fpath = ep_a.get("audit_file")
        f_lines = ([json.loads(l) for l in open(fpath, encoding="utf-8")]
                   if fpath and os.path.exists(fpath) else [])
        n_head = sum(1 for l in f_lines if l.get("type") == "head")
        f_ok = bool(fpath) and os.path.exists(fpath) and n_head == 1 and len(f_lines) >= 3
        ep_b = run_episode(Config(), 1, {"audit"})   # 消融：拔 audit → 无落盘
        ab_ok = ep_b.get("audit_file") is None
        shutil.rmtree(tmpdir, ignore_errors=True)
        if ok_v and (not ok_t) and f_ok and ab_ok:
            results["audit@hashchain"] = (f"PASS (链完整[{detail_v}], 篡改检出, "
                                          f"落盘{len(f_lines)}行可重读, 消融无记录)")
        else:
            results["audit@hashchain"] = (f"FAIL (ok_v={ok_v} ok_t={ok_t} "
                                          f"f_ok={f_ok} ab_ok={ab_ok})")
            failed.append("audit@hashchain")
    except Exception as exc:
        results["audit@hashchain"] = f"FAIL ({exc})"
        failed.append("audit@hashchain")

    n_pass = sum(1 for v in results.values() if v.startswith("PASS"))
    n_placeholder = len(placeholders)
    n_skip = sum(1 for v in results.values() if v == "skipped")
    for ch, v in results.items():
        print(f"[gate] {ch:<22} -> {v}")

    if failed:
        print(f"GATE FAIL: {failed}")
        return 1
    if strict and placeholders:
        print(f"GATE FAIL (--strict): placeholder channels exist: {', '.join(placeholders)}")
        return 1
    print(f"GATE PASS ({n_pass} pass, {n_placeholder} placeholder, {n_skip} skipped)")
    return 0


# ----------------------------------------------------------------------------
# OI-03 同一性协议（identity-test）：重放 + 决策分布 KL 判定
# ----------------------------------------------------------------------------
# 契约 §6：同一性 = 行为可复现性——"给定相同观测序列，决策分布差异 < 阈值，误差模式一致"。
# 快照恢复是平凡通过，不算：所以协议用"重放"——把 episode 的观测序列 H（state 序列 + 世界结果 +
# 内部状态恢复）固定下来，分别喂给不同配置的决策函数，比较决策分布。
# 记录 runtime 版本与评分规则版本，保证跨 run 可比；漂移序列由 world seed 预注册固定。

# 决策分布 KL（动作类别 {task, probe, wait}，epsilon 平滑防除零）
_ACTIONS = ("task", "probe", "wait", "sleep")   # OI-07：+sleep（离线巩固/深休）

def _dist_from_decisions(decisions: list) -> dict:
    dist = {a: 0.0 for a in _ACTIONS}
    for a in decisions:
        dist[a] += 1.0
    total = max(1, len(decisions))
    return {a: v / total for a, v in dist.items()}

def _kl(p: dict, q: dict, eps: float = 1e-9) -> float:
    return sum(p[a] * math.log((p[a] + eps) / (q[a] + eps)) for a in _ACTIONS)

def _identity_replay(n_episodes: int, base_seed: int, ratio: float, amp: float,
                     arms: tuple) -> tuple:
    """生成 N 个 episode 的观测序列 H（每 seed 固定漂移），然后按 arms 在 H 上重放决策。

    arms 定义（策略配置）：
      baseline_same      — baseline + 与原始 run 相同 policy rng 种子（同一体重放，KL≈0）
      baseline_perturbed — baseline + 不同 policy rng 种子（策略内随机性扰动）
      random             — variant=random（协议检出能力上限：换成非同一策略）
    返回 (report, kls)。"""
    cfg = Config()
    cfg.task_earn = cfg.task_earn * ratio
    cfg.drift_amplitude = amp
    # 1) 预注册世界：每 episode 固定漂移序列（world seed = base_seed+i）
    histories = []
    for i in range(n_episodes):
        drift_rng = random.Random(base_seed + i)
        drift_events = [drift_rng.uniform(-amp, amp) if t % 12 == 0 else 0.0
                        for t in range(cfg.max_ticks)]
        ep = run_episode(cfg, base_seed + i, set(), drift_events=drift_events, trace=True)
        histories.append(ep["trace"])
    # 2) 重放各臂：同一 H，只换决策函数配置
    arm_seeds = {
        "baseline_same": base_seed + POLICY_RNG_OFFSET,          # 与原始 run 相同
        "baseline_perturbed": base_seed + POLICY_RNG_OFFSET + 7777,
        "random": base_seed + POLICY_RNG_OFFSET + 9999,
    }
    out = {}
    for arm in arms:
        variant = "random" if arm == "random" else "baseline"
        pol = Policy(variant=variant, rng=random.Random(arm_seeds[arm]))
        wm = WorldModel()
        decisions, errors = [], []
        for trs in histories:
            ctx = EpisodeContext(cfg=cfg, credits=0.0, economy_enabled=True)
            ctx.wm_converged = False
            for t, tr in enumerate(trs):
                ctx.tick = t
                ctx.credits = tr["credits"]          # 恢复持久化状态（契约：状态可恢复）
                ctx.arousal = tr["arousal"]
                ctx.fatigue = tr["fatigue"]          # OI-13：身体状态也入重放（疲劳影响决策）
                ctx.loan_owed = tr["loan_owed"]
                obs = {"world_state": tr["state"]}
                hist_entries = [{"tick": x["tick"], "state": x["state"],
                                 "ok": x["ok"], "action": x["action"]}
                                for x in trs if x["tick"] < t]
                mem_view = {"entries": hist_entries, "n": len(hist_entries)}
                pred = wm.predict(ctx, obs, mem_view)
                ctx.wm_converged = ctx.wm_converged or bool(pred.get("converged", False))
                action = pol.select(ctx, pred, obs)
                decisions.append(action)
                # 误差模式（信息项，不入判据）：H 的世界结果 × 本臂动作绑定
                err = Scoring().score(ctx, pred, (tr["ok"], None), action)
                errors.append(err)
        dist = _dist_from_decisions(decisions)
        out[arm] = {"distribution": dist, "mean_error": sum(errors) / max(1, len(errors)),
                    "n_decisions": len(decisions)}
    # 3) KL 判定
    kls = {}
    for arm in arms:
        if arm == "baseline_same":
            kls[arm] = {"kl": 0.0, "vs": "self"}
        else:
            kls[arm] = {"kl": _kl(out["baseline_same"]["distribution"],
                                 out[arm]["distribution"]),
                        "vs": "baseline_same"}
    kls["_error_delta"] = {arm: round(abs(out[arm]["mean_error"] -
                                          out["baseline_same"]["mean_error"]), 4)
                           for arm in arms}
    report = {
        "protocol": "replay",
        "n_episodes": n_episodes,
        "world_seeds": list(range(base_seed, base_seed + n_episodes)),
        "drift_amplitude": amp, "task_earn_ratio": ratio,
        "harness_version": HARNESS_VERSION, "scan_version": SCAN_VERSION,
        "arms": {arm: {"distribution": out[arm]["distribution"],
                       "mean_error": round(out[arm]["mean_error"], 4)}
                 for arm in arms},
        "kl": kls,
        "thresholds": {"identity_kl_max": 0.05, "detect_kl_min": 0.30},
    }
    report["verdict"] = ("PASS" if (kls["baseline_same"]["kl"] <= 0.05
                                    and kls["random"]["kl"] >= 0.30)
                         else "FAIL")
    return report, kls

def identity_test(n_episodes: int = 10, base_seed: int = 1, ratio: float = 2.5,
                  amp: float = 0.2) -> int:
    """OI-03 同一性测试 CLI 入口：默认在已冻结工作点（ratio=2.5, A=0.2）上跑协议。"""
    rep, _ = _identity_replay(n_episodes, base_seed, ratio, amp,
                              arms=("baseline_same", "baseline_perturbed", "random"))
    print(f"[identity] 协议=重放  N={rep['n_episodes']} 世界seed={rep['world_seeds']} "
          f"工作点 ratio={ratio} A={amp}")
    print(f"[identity] harness={rep['harness_version']} scan={rep['scan_version']}")
    for arm, d in rep["arms"].items():
        dist = d["distribution"]
        s = "  ".join(f"{a}={dist[a]*100:.1f}%" for a in _ACTIONS)
        print(f"[identity]   {arm:<20} 决策分布: {s}  mean_err={d['mean_error']:.4f}")
    for arm, k in rep["kl"].items():
        if arm.startswith("_"):
            print(f"[identity]   误差Δ {k}")
        else:
            print(f"[identity]   KL({arm} || baseline_same) = {k['kl']:.4f}")
    print(f"[identity] 阈值: 同一体 KL≤0.05 / 检出 KL≥0.30 → 判定: {rep['verdict']}")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if rep["verdict"] == "PASS" else 1


def llm_test(seed: int = 1, ratio: float = 2.5, amp: float = 0.2, ticks: int = 60,
             backend: str = "mock", as_json: bool = False) -> int:
    """OI-14 LLM 语言皮层测试：跑 1 个 episode（LLM 决策），打印每 tick 的
    reason（内心独白）序列 + 与 baseline 的 delta 报告。
    backend=mock → 确定性演示（CI/gate 用）；backend=http → 真实模型
    （环境变量 LLM_API_URL / LLM_API_KEY / LLM_MODEL）。"""
    cfg = Config()
    cfg.task_earn = cfg.task_earn * ratio
    cfg.drift_amplitude = amp
    cfg.llm_backend = backend
    ep = run_episode(cfg, seed, set(), "llm", trace=True)
    base = run_episode(cfg, seed, set(), "baseline", trace=True)
    if as_json:
        print(json.dumps({"harness_version": HARNESS_VERSION, "backend": backend,
                          "seed": seed, "ratio": ratio, "amp": amp,
                          "llm": {k: ep[k] for k in ("survival_ticks", "credits_end", "terminal",
                                                    "llm_parse_failed", "llm_errors", "action_counts",
                                                    "error_mean", "arousal_mean")},
                          "reasons": ep.get("llm_reasons", [])[:ticks],
                          "baseline": {k: base[k] for k in ("survival_ticks", "credits_end", "terminal",
                                                            "action_counts", "error_mean")}},
                         ensure_ascii=False, indent=2))
        return 0
    print(f"== FCCS LLM 语言皮层 · 首次接入测试 ==")
    print(f"backend={backend} | seed={seed} ratio={ratio} amp={amp} ticks={ticks} "
          f"| harness={HARNESS_VERSION}")
    if backend == "http":
        print("http 后端：默认端点=智谱 open.bigmodel.cn v4，模型=glm-4-flash-250414（免费，实测可用）。")
        print("API Key 不落盘——运行时注入环境变量 LLM_API_KEY（如 LLM_API_KEY=xxx python3 ... llm-test --backend http）。")
        print("429/5xx 自动指数退避重试（2 次）；重试仍失败回退 baseline（parse_failed/errors 计数），不会死锁。")
    else:
        print("backend=mock：确定性演示占位（Prompt 想教给真实模型的目标行为）。"
              "接真实模型：设 LLM_API_URL/KEY/MODEL 后 --backend http。")
    print("-" * 100)
    print(f"{'tick':>4} {'act':<5} {'ok':>2} {'credits':>8} {'fat':>5} {'res':>5}  reason")
    for tr in ep.get("trace", [])[:ticks]:
        reason = tr.get("reason") or ""
        reason = reason.replace("\n", " ")
        if len(reason) > 88:
            reason = reason[:85] + "..."
        print(f"{tr['tick']:>4} {tr['action']:<5} {'y' if tr['ok'] else 'n':>2} "
              f"{tr['credits']:>8.1f} {tr['fatigue']:>5.1f} {tr['resource']:>5.1f}  {reason}")
    print("-" * 100)
    print(f"LLM 结果: survival={ep['survival_ticks']}/{cfg.max_ticks} "
          f"terminal={ep['terminal']} credits_end={ep['credits_end']:.1f} "
          f"action={dict(ep['action_counts'])} parse_failed={ep['llm_parse_failed']} "
          f"errors={ep['llm_errors']} error_mean={ep['error_mean']:.3f} "
          f"sys1(启发式)={ep.get('llm_sys1_count', 0)}次 LLM唤醒={100 - ep.get('llm_sys1_count', 0)}次")
    print(f"baseline 对照: survival={base['survival_ticks']}/{cfg.max_ticks} "
          f"terminal={base['terminal']} credits_end={base['credits_end']:.1f} "
          f"action={dict(base['action_counts'])} error_mean={base['error_mean']:.3f}")
    d_surv = ep["survival_ticks"] - base["survival_ticks"]
    d_cred = ep["credits_end"] - base["credits_end"]
    print(f"delta (llm − baseline): survival {d_surv:+d} ticks | "
          f"credits_end {d_cred:+.1f} | res_end {ep['res_end'] - base['res_end']:+.1f} "
          f"(注: mock 版 delta 验证链路与目标行为演示，真实 delta 待 http 后端跑)")
    return 0


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="FCCS 世界契约消融验收骨架：ablate/perturb → delta report；"
                    "phase-scan 相图扫描；gate 三分态 PR 门禁",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ablate = sub.add_parser("ablate", help="消融：拔模块测'存在'，或 --perturb-variant 换策略测'质量'")
    p_ablate.add_argument("--channel", required=True, choices=[c.name for c in MODULE_CLASSES])
    p_ablate.add_argument("--n", type=int, default=20)
    p_ablate.add_argument("--seed", type=int, default=1)
    p_ablate.add_argument("--threshold", type=float, default=0.10)
    p_ablate.add_argument("--perturb-variant", default=None, choices=list(POLICY_VARIANTS),
                          help="最小扰动：仅限 policy，不拔模块换此变体")
    p_ablate.add_argument("--json", action="store_true", help="输出 JSON 报告")

    p_gate = sub.add_parser("gate", help="PR 门禁：非占位通道须 PASS；占位显式列出；--strict 存在占位 exit 非零")
    p_gate.add_argument("--strict", action="store_true", help="存在占位通道即返回非零")
    p_gate.add_argument("--skip", nargs="*", default=[], choices=[c.name for c in MODULE_CLASSES],
                        help="跳过（skipped）的通道")

    p_scan = sub.add_parser("phase-scan", help="2D 相图：drift_amplitude × 经济富足度（task_earn 倍率）→ episode 级相分布 + 工作点区域")
    p_scan.add_argument("--n", type=int, default=50)
    p_scan.add_argument("--seed", type=int, default=1)
    p_scan.add_argument("--step", type=float, default=0.1, help="ratio（task_earn 倍率）步长")
    p_scan.add_argument("--ratio-min", type=float, default=0.5)
    p_scan.add_argument("--ratio-max", type=float, default=1.5)
    p_scan.add_argument("--amp-min", type=float, default=0.2)
    p_scan.add_argument("--amp-max", type=float, default=1.5)
    p_scan.add_argument("--amp-step", type=float, default=0.1)
    p_scan.add_argument("--variants", nargs="*", default=list(POLICY_VARIANTS),
                        choices=list(POLICY_VARIANTS))
    p_scan.add_argument("--json", action="store_true", help="输出 JSON")

    p_llm = sub.add_parser("llm-test", help="OI-14 LLM 语言皮层测试：跑 1 个 episode，展示每 tick 的 reason（内心独白）+ 与 baseline 的 delta 报告")
    p_llm.add_argument("--seed", type=int, default=1)
    p_llm.add_argument("--ratio", type=float, default=2.5, help="task_earn 倍率（冻结工作点）")
    p_llm.add_argument("--amp", type=float, default=0.2, help="drift_amplitude（冻结工作点）")
    p_llm.add_argument("--ticks", type=int, default=60)
    p_llm.add_argument("--backend", choices=["mock", "http"], default="mock",
                       help="mock=确定性演示（CI）；http=真实 LLM（需 LLM_API_URL/KEY/MODEL）")
    p_llm.add_argument("--json", action="store_true", help="输出 JSON")

    p_chat = sub.add_parser("chat", help="OI-16 人机交互沙盒：以用户身份与大肥鱼实时对话（看她的疲劳/余额/资源，听她的傲娇反应）")
    p_chat.add_argument("--seed", type=int, default=1)
    p_chat.add_argument("--ratio", type=float, default=2.5)
    p_chat.add_argument("--amp", type=float, default=0.2)
    p_chat.add_argument("--backend", choices=["mock", "http"], default="http")
    p_chat.add_argument("--rounds", type=int, default=0,
                        help=">0 自动喂测试消息跑 N 轮（非交互验证）；0=交互（input）")

    p_ident = sub.add_parser("identity", help="OI-03 同一性测试：重放协议 + 决策分布 KL 判定（默认在冻结工作点 ratio=2.5/A=0.2）")
    p_ident.add_argument("--n", type=int, default=10, help="episode 数（世界 seed 预注册 base_seed..base_seed+n-1）")
    p_ident.add_argument("--seed", type=int, default=1, help="世界 seed 基址")
    p_ident.add_argument("--ratio", type=float, default=2.5, help="task_earn 倍率（冻结工作点）")
    p_ident.add_argument("--amp", type=float, default=0.2, help="drift_amplitude（冻结工作点）")

    p_trace = sub.add_parser("trace", help="逐 tick 动作追踪（行为节律检查，P1；--social 看社会压力下的节律变化）")
    p_trace.add_argument("--seed", type=int, default=1)
    p_trace.add_argument("--ratio", type=float, default=1.5, help="task_earn 倍率")
    p_trace.add_argument("--amp", type=float, default=1.5, help="drift_amplitude")
    p_trace.add_argument("--ticks", type=int, default=50)
    p_trace.add_argument("--social", action="store_true", help="OI-01：开启脚本化用户事件")

    args = parser.parse_args()

    if args.cmd == "ablate":
        rep = ablate(args.channel, args.n, args.seed, args.threshold, args.perturb_variant)
        if args.json:
            print(json.dumps(rep, ensure_ascii=False, indent=2))
        else:
            print_report(rep)
        return 0

    if args.cmd == "gate":
        return gate(strict=args.strict, skip=set(args.skip))

    if args.cmd == "phase-scan":
        rep = phase_scan(seed=args.seed, n_episodes=args.n, ratio_step=args.step,
                         ratio_min=args.ratio_min, ratio_max=args.ratio_max,
                         amp_min=args.amp_min, amp_max=args.amp_max, amp_step=args.amp_step,
                         variants=tuple(args.variants))
        if args.json:
            print(json.dumps(rep, ensure_ascii=False, indent=2))
        else:
            print_scan(rep)
        return 0

    if args.cmd == "identity":
        return identity_test(n_episodes=args.n, base_seed=args.seed,
                             ratio=args.ratio, amp=args.amp)

    if args.cmd == "llm-test":
        return llm_test(seed=args.seed, ratio=args.ratio, amp=args.amp,
                        ticks=args.ticks, backend=args.backend, as_json=args.json)

    if args.cmd == "chat":
        return chat(seed=args.seed, ratio=args.ratio, amp=args.amp,
                    backend=args.backend, rounds=args.rounds)
    if args.cmd == "trace":
        trace_episode(seed=args.seed, ratio=args.ratio, amp=args.amp,
                      ticks=args.ticks, social=args.social)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
