# FCCS 最小循环伪代码 + 模块接口表（含消融钩子）v0.3

> 关联：`world_contract.md` v0.1（§0.2 最小循环、§7 审计与消融）
> 配套：`ablation_harness.py` v0.3.0（模块名/版本/消融通道与本表一一对应）
> 硬约束：**每个模块必须有 `enabled` + `version`；无 LLM 基线可跑；实现阶段每个 PR 必须挂 harness 跑通，否则不合并。**

## 变更记录

- **v0.2（P0/P1 修复，对应 harness v0.3.0）**
  - P0-1 obs 进入世界模型：`world_model` 按 state 分桶学习 p_ok；`policy` 至少读 obs 一个字段。
  - P0-2 `world.state` 调制执行失败率：`fail_p = base + g(state)`；记忆条目含 state。
  - P0-3 `prediction_log` 不可变：append 深拷贝；scoring 只对 `pred_log[-1]` 计分；`pred_hash`（sha256）进日志与审计事件。
  - P0-4 感知计价：probe 成本 = 通道 × depth，走同一本账。
  - P1-5 economy 消融时 physiology 扣费 no-op（无限资源不再 starved）。
  - P1-6 policy 消融双臂：ablate（存在）+ perturb:random（质量），gate 双臂必跑。
  - P1-7 审计事件补 `cause_class` / `enabled_channels` / `module_versions` / `pred_hash`。
- **v0.3（P0-2 轮，对应 harness v0.4.0）**
  - 感知计价与行动解耦：感知每 tick 无条件计费（`sense_cost_per_depth × depth`），与行动计价分离。
  - perturb 配对干净：Policy 与 World 独立 rng 流（`policy_rng = seed + POLICY_RNG_OFFSET`），两臂 world 随机流一致。
  - `World.advance` off-by-one 修正（漂移按 tick 边界应用，tick 0 不漂移；补回循环调用）。
  - `_scan_metrics` 分母改用实际总行动数。
  - pred_log 唯一权威：scoring 只读 `ctx.pred_log[-1]`；`audit.prediction_log` 为镜像（持久化哈希链）。

---

## 1. 最小循环（正式伪代码）

```
LOOP（每 tick）:
  0.  ECONOMY.supply(); ECONOMY.metabolism()          # 基础补给 < 维持成本，每 tick 净流出
      IF credits <= 0 → TERMINAL = starved
  1.  mem ← MEMORY.read()                             # 记忆读进循环头；条目含 state（P0-2）
  2.  probe ← ATTENTION.select(mem, arousal, budget)  # {channel, depth, budget}；depth 计价（P0-4）
  3.  obs ← PERCEPTION.sense(world, probe)            # obs 含 world_state（精度随 depth）
  4.  pred ← WORLD_MODEL.predict(obs, mem)            # obs→p_ok：state 分桶学习（P0-1）
      payload ← deepcopy(pred)                        # 唯一权威副本（P0-3）
      ctx.pred_log.append(payload)                    # scoring 唯一读取源
      pred_hash ← AUDIT.commit_prediction(payload)    # 镜像：Runtime 独占写哈希链（P0-3）
  5.  action ← POLICY.select(pred, obs, arousal, budget)  # 至少读 obs 一个字段（P0-1）
  6.  result ← ACTION_EXECUTOR.execute(action)        # fail_p = base + g(state)（P0-2）
  7.  error ← SCORING.score(pred_log[-1], result)     # 只对已提交副本计分（P0-3）
  8.  arousal ← PHYSIOLOGY.update(error)              # 扣费受 economy_enabled 守卫（P1-5）
  9.  social_arousal ← SOCIAL.update(user_events)     # 独立唤醒轴，低权重
 10.  MEMORY.write(obs, result, error)                # 条目含 state（P0-2）
 11.  ECONOMY.settle(action, probe)                   # 感知每 tick 无条件计价（sense×depth）；行动计价分离（P0-2）
      IF credits <= 0 → TERMINAL = starved
 12.  AUDIT.record(event)  # 含 pred_hash / cause_class / enabled_channels / module_versions（P1-7）
 13.  world.advance(tick)                             # 非平稳漂移（off-by-one 修正：tick 边界应用）

TERMINAL ∈ {max_ticks, starved, crashed}
```

不变式：
- 预测先提交、后打分；提交即封存（深拷贝 + 哈希），不可修改、不可删除；LLM/智能体无写权限。
- 所有限制以"预算耗尽 / 行动失败"返回，不以静默压制表达。
- 内部思考也计价（token/推理步数）；元认知 = 预算分配问题。

---

## 2. 模块接口表（含消融钩子）

| ID | 模块 | 接口 | 输入 → 输出 | 计价 | 失败模式 | 消融钩子（enabled=False 的中性默认） |
| --- | --- | --- | --- | --- | --- | --- |
| M1 | attention | `select(mem, arousal, budget) → probe{channel, depth}` | 记忆/唤醒/预算 → 探测计划 | 感知按通道×深度 | 探测失败（网络） | 固定探测（ch0，depth=1，最便宜） |
| M2 | perception | `sense(world, probe) → obs` | 探测计划 → 部分观测（含 state） | 通道×深度 | 超时/失败 | 返回空观测（无 state → 世界模型无信息） |
| M3 | memory | `read() → (entries, n)`；`write(obs, result, error)`；`forget()` | 观测/结果/误差 → 记忆更新 | 存储成本；遗忘有性能损失 | 存储满 | read→空；write/forget→no-op |
| M4 | world_model | `predict(obs, mem) → pred_dist`（commit） | 观测 state/记忆 → 预测分布 | 思考计价 token/步 | commit 被拒（日志完整性） | 提交均匀分布（无信息 → 恒 log2 误差） |
| M5 | policy | `select(pred, obs, arousal, budget) → action` | 预测/观测/唤醒/预算 → 行动（task/probe/wait/**sleep**，v0.13.0） | 思考计价 | — | 返回 wait（最便宜动作） |
| M6 | action_executor | `execute(action) → result{ok, cause}` | 行动 → 结果；成功 task 消耗世界资源（World Mutation，§1.3） | 延迟+配额 | 超时/配额耗尽/写入失败/网络抖动；fail_p = base+g(state)；sleep 恒成功 | 返回成功（无操作） |
| M7 | scoring | `score(pred_log[-1], result) → error` | 已提交预测+结果 → 误差 | — | 日志不可用 | 误差恒 0（无学习信号） |
| M8 | physiology | `update(error) → arousal`；表达预算耦合 | 误差/资源 → 唤醒 | 唤醒烧 credits/tick（economy_enabled 守卫） | 觉醒地板 | 唤醒恒 0 |
| M9 | economy | `supply()/metabolism()/settle(action, probe)/earn()` | 计价事件 → 账本更新 | 感知每 tick 无条件计价（sense×depth）+ 行动计价 | 透支 → starved | 不计价（无限资源；生理扣费同步 no-op） |
| M10 | social | `update(user_events) → social_arousal` | 用户行为事件 → 社会唤醒轴 | — | 事件缺失 | 社会轴恒 0 |
| M11 | audit | `record(event)`；`commit_prediction(payload) → pred_hash` | 全事件 → append-only 日志 + prediction_log 镜像 | 写日志成本 | 日志写失败 | 不记录、不提交 |

### 约定

- **版本语义**：模块实现逻辑变更必须 bump 该模块 `version`（语义化），并在 PR 描述说明变更。
- **enabled 语义**：`ablate(channel)` 即该模块 `enabled=False`；消融时使用表中"中性默认"替换其因果贡献，**不删除模块、不破坏循环结构**。
- **policy 消融双臂（P1-6）**：
  - `ablate --channel policy` = 拔模块（enabled=False → wait），测**存在性**；
  - `ablate --channel policy --perturb-variant random` = 不拔模块、换随机策略，测**质量**（最小扰动）。
  - 两张 delta 报告都是 policy 变更 PR 的必备证据。
- **消融判据**：拔掉模块 X 后行为分布显著变化（delta ≥ 阈值）。占位通道（当前 social/audit）显式列出、不计入 pass；落地后由 gate 复测。
- **无 LLM 基线**：world_model = state 分桶频率估计（无数据→均匀分布）；policy = 启发式三变体（baseline/random/arousal_aware）。全部确定性实现，harness 在任何环境可跑。

---

## 3. 审计事件与 prediction_log schema

### 3.1 prediction_log（Runtime 独占写，append-only 哈希链）

```
prediction_log 条目 = {
  seq,                        # 递增序号
  tick,
  pred_hash,                  # sha256(deepcopy(pred) 序列化)，不可变指纹
  prev_hash,                  # 前一条 pred_hash（哈希链，防改写历史）
  pred,                       # 深拷贝的已提交预测分布
}
```

- **唯一权威**：运行时权威 = `ctx.pred_log`（scoring 只读 `ctx.pred_log[-1]`）；`audit.prediction_log` 仅作持久化镜像，不参与评分。
- **写入权限**：仅 `AUDIT.commit_prediction`（Runtime 侧）可写；LLM/智能体无写权限、无修改权限、无删除权限。
- **不可变**：append 时深拷贝 + 哈希；提交后不可改、不可删。
- 事后补预测、改历史、对冲（提交宽分布）均被评分规则本身计价（契约 §1.5），与日志完整性正交。

### 3.2 审计事件

```
event = {
  tick,
  action, ok, cause_class,     # cause_class ∈ {timeout, network, quota, ...}；自致/外生判定由规则完成
  error, arousal, social_arousal,
  pred_hash,                   # 关联本条 prediction_log 条目
  credits_after,
  enabled_channels,            # 全模块 enabled 快照（消融可审计）
  module_versions,             # 全模块 version 快照（同一性回放可比性）
  terminal,
}
```

---

## 4. PR 门禁（硬规则）

**合并条件（三者齐才合并）**：

1. `python ablation_harness.py gate`：**所有非占位通道 PASS**（含 policy 双臂：ablate + perturb:random）。
2. **占位通道显式列出**：gate 输出区分 pass / placeholder / skipped；占位清单（当前：social、audit）写在 PR 描述。
3. **变更模块 delta 报告**：`ablate --channel X`（policy 另附 `--perturb-variant random`）附在 PR 描述。

其余规则：

- PR 描述必须写明 **"契约机制落实率"**：本次 PR 后已落实机制数 / 契约机制总数，并列出剩余占位与 TODO(impl)。
- 任一模块逻辑变更必须 bump 该模块 `version`。
- 新增模块必须：进本表 + harness `MODULE_CLASSES` + 提供 `enabled`/`version`/消融钩子；占位必须进 `PLACEHOLDER_CHANNELS` 显式声明。
- `python ablation_harness.py gate --strict`：存在占位通道即 exit 非零（完整度门禁，用于收尾验收）。
- gate 不满足 → 不合并。

---

## 5. 与 world_contract.md 的对应

| 契约章节 | 落点 |
| --- | --- |
| §0.2 最小循环 | 本文件 §1 |
| §1.3 世界非平稳 | World（state 漂移 + fail_p = base+g(state)） |
| §1.4 稀缺经济 / 相图 | M9 + `phase-scan`（已实现：比值 × policy 变体 → 三相 + 相变点） |
| §1.5 评分规则 | M7 + M4（commit）+ prediction_log（§3.1） |
| §1.7 觉醒地板/恢复 | M8（TODO 实现） |
| §1.9 探索红利 | M5（待定决策） |
| §2 观测接口 | M1/M2（感知计价 = 通道×depth） |
| §3 行动接口 | M5/M6 |
| §4 代价与延迟 | M9 + 循环第 0/11 步 |
| §5 失败模型 | M6 + World.execute |
| §6 持久化/同一性 | M3 + `identity`（占位） |
| §7 审计与消融 | M11 + harness `ablate/gate` |

---

## 6. 经济工作点（phase-scan v0.2.0 结论，可复核）

扫描命令：`python ablation_harness.py phase-scan --n 100 --seed 1 --ratio-min 0.1 --ratio-max 2.0`
（固定漂移 seed=1；维护 = 2.0/tick；感知 = 1.0/tick 无条件计价；步长 0.1，每点 100 episode）。

**相变点（按策略）**：

| 策略 | panic→homeostasis | homeostasis→meditation | 备注 |
| --- | --- | --- | --- |
| baseline | 0.7 | 1.1 | 参考策略 |
| random | 1.3 | 永不冥想 | surplus 下仍烧资源（wait≈33%）——策略质量可测 |
| arousal_aware | 0.7 | 1.5 | 唤醒驱动探测有成本，比基线晚冥想 |

**推荐工作点：ratio = 1.0（supply = maintenance = 2.0）**
- 维持成本恰被补给覆盖，但感知成本（1.0/tick）构成真实赤字 → 必须靠任务赚取差额（task 率 ≈ 36%）。
- 实测：存活 94.8 tick / 死亡 9% / wait 63.7% —— 处于内稳态相、有真实稀缺压力、不恐慌。
- 等价表述：总维持 = 维持 + 感知 = 3.0，补给/总维持 = 0.67，符合契约"补给 < 维持"。
- 再校准：要更紧张（更高 task 率）取 0.8–0.9；要更低死亡取 1.0–1.05。

**纳入 PR 门禁**：经济参数变更 PR 必须附更新后的 phase-scan 报告（相变点 + 工作点），不得只改数字不重扫。
