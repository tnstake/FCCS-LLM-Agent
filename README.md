# FCCS — Functional Consciousness Candidate System

**A High-Constraint Survival Sandbox for LLM Agents.**

> Consciousness is not something you *give* an agent — it is something an agent *earns* by surviving a world that does not care.

FCCS is a minimal, fully deterministic simulation harness that drops a language-model agent into a **negative-sum, non-stationary, resource-scarce world** and asks one question: under which physiological and economic constraints does an agent develop *adaptive survival behavior* — probing, working, resting, and even showing restraint — instead of collapsing into either mindless toil or fatalistic passivity?

The system is designed around a **World Contract**: a fixed set of mechanisms (economy, physiology, perception, social feedback, auditability) that the agent cannot cheat or rewrite. The harness ships with a fully scripted baseline (no LLM), a real-LLM policy backend, a dual-process (System 1 & 2) decision architecture, a natural-language **Persona** layer, a 2-D phase-diagram scanner, ablation/identity test suites, and a CI-style PR gate — **26/26 assertions passing, zero placeholders**.

---

## Key Ideas

### 1. Negative-Sum Economy (负和经济)
The world never gives you enough to survive on. `supply_per_tick < maintenance_per_tick` — every tick bleeds credits. Perception is **unconditionally billed per tick** (channel × depth), independent of action. A startup loan exists but must be repaid from task earnings. There is **no free-meditation attractor**: idling is a *net outflow*, and wealth can only be earned through risky action. Curiosity is not free either — probing is priced by an explicit **Value-of-Information** accounting (`VoI = uncertainty × expected surplus × remaining horizon − cost`).

### 2. Fatigue & Physiology (疲劳与生理机制)
Arousal is driven by prediction error through a fixed, versioned physiological cost table — error accumulates arousal, arousal burns credits. Arousal decays back to baseline (never saturates), but the cost rate never decreases. **Fatigue** breaks the perpetual-motion bug: working and probing tire the body, failure rates rise with fatigue, and only `wait`/`sleep` recover it. Sleep is not a free shelter — basal metabolism still burns, and advanced perception is halved. On waking, **offline consolidation** (`consolidate()`) distills `error > threshold` episodes into a *permanent Bayesian prior* — the agent literally develops long-term trauma ("early childhood experience") that no sliding window can erase.

### 3. World Mutation — Actions Leave Scars (世界反作用力)
The world is not a passive stage. Every successful `task` depletes world resources; resource depletion **amplifies negative drift** — over-exploitation turns the environment hostile, and success rates collapse. Restore the resource by restraint. This forces a real economic-ethics trade-off: **burn the world down, or learn to live in it**. The frozen working point shows the winning policy is not the most aggressive one.

### 4. Dual-Process System 1 & 2 (快慢双系统)
A naive "LLM decides every tick" policy (System 1 only) *fails*: constant survival-cost stimuli induce tunnel vision, reckless overwork, and near-bankruptcy (70% survival). FCCS separates cognition:
- **System 1** — the scripted heuristic core handles routine ticks (fatigue-aware resting, p_ok-gated working, EV-based probing);
- **System 2** — the LLM is woken every `llm_call_interval = 10` ticks (or on credit emergency) to make *strategic* plans with full state visibility, including explicit stop-loss rules and world-resource awareness.

Result: survival 70% → **90%**, prediction error *lower* than the heuristic baseline, and world resources preserved instead of strip-mined.

### Additional machinery
- **Independent social axis** — user feedback (positive/negative/supply-interruption) drives a *separate* social-arousal channel at 0.30× weight, never mixed with physical error; scripted user events are gated off by default so scans stay frozen-comparable.
- **Audit hashchain** — every prediction and event is appended to an append-only SHA-256 hash chain, written exclusively by the runtime (the LLM has no write permission). Tamper-evident, JSONL-persistable.
- **Persona layer** — a pure natural-language expression interface (e.g., the tsundere "Big Fat Fish") that *speaks* but never *decides*; decision architecture untouched.
- **Identity protocol** — same-world replay with KL-threshold decision distributions: identical config ⇒ identical decisions (Δ = 0.0000), perturbed config ⇒ detected (Δ = 0.0005), random policy ⇒ rejected.

---

## Architecture

```
                 ┌────────────────────────────────────────────────────┐
   user events   │  System 1 (heuristic)   ── every tick              │
  (social axis)  │  System 2 (LLM)         ── every 10 ticks / panic  │
                 │  Persona (expression)   ── speaks, never decides   │
                 └───────────────────────┬────────────────────────────┘
                                         │ action
        World (non-stationary, mutable) ◄┘ outcome
                 │ state / success / error
        Scoring (proper scoring, commit-before-observe)
        Physiology (arousal + fatigue) → burns credits
        Economy (negative-sum, billed perception, loan)
        Memory / WorldModel (action-conditional Beta estimation)
        Social (independent arousal axis)
        Audit (append-only hashchain)   ── runtime-exclusive write
```

Each module is a class with `name / version / enabled`; disabling it is a **one-line ablation** that yields a behavior delta. The 13-step per-tick loop is fixed and audited (see `loop_pseudocode.md`).

| Module | Version | Role |
|---|---|---|
| `World` | — | Non-stationary drift (clamped mean-reverting), world-resource ledger, state-modulated success |
| `Attention` | 0.5.0 | Probe decisions priced by Value-of-Information; arousal-driven deep sensing |
| `WorldModel` | 0.4.3 | Action-conditional Beta(2,2) estimates, sliding window, belief decay, permanent consolidated priors |
| `Policy` / `LLMPolicy` | 2.0.0-llm | System 1 heuristic core / System 2 strategic LLM (mock or HTTP backend) |
| `Persona` | 0.1.0 | Natural-language expression layer (character injection) |
| `Physiology` | 0.4.0 | Fixed error→arousal→credit cost table; fatigue dynamics |
| `Economy` | 0.4.0 | Negative-sum ledger; unconditional per-tick perception billing; repayable loan |
| `Social` | 0.2.0 | Independent social-arousal axis, user-behavior memory, supply interruption |
| `Audit` | 0.4.0 | Append-only SHA-256 hashchain; JSONL persistence; tamper detection |

---

## Results

All experiments use the **frozen economic working point** `ratio 2.5 × drift_amplitude 0.2` (`task_earn = 18.75`), 100-tick episodes, fixed drift sequence per seed, `max_ticks=100`. LLM backend: `glm-4-flash-250414` (free tier), temperature 0.8.

### Survival & behavior — 10-episode evaluation (seeds 1–10, real LLM)

| Variant | Survival | Task rate | Avg credits end | Avg world-resource end | Mean pred. error |
|---|---|---|---|---|---|
| Scripted baseline (System 1 only) | survives\* | ~50% | — | **0.0** (strip-mined) | 0.352 |
| Pure-LLM per-tick (naïve) | **7/10** | **87%** | 106.6 | 6.1 | 0.561 |
| **Dual-process (System 1 & 2)** | **9/10** | 44% | **127.6** | **7.5** | **0.319** |

\* Baseline's survival is qualitative: as a scripted policy it always completes 100-tick episodes at the frozen work point, but it exhausts the world resource (`res_end = 0.0`) and shows no restraint — it trades the long term for short-term income. Survival rates (7/10, 9/10) are from 10-episode evaluations (seeds 1–10, real LLM); prediction-error column: dual-process = 10-episode mean, baseline & pure-LLM = same-config seed-5 reference.

**The three headline findings:**
1. **Pure-LLM per-tick decision-making is actively harmful** — 70% survival, 87% overwork, near-bankruptcy. Constant survival-cost stimuli produce tunnel vision (the "loan-crazed gambler" pathology).
2. **Dual-process reasoning fixes it** — 90% survival, task rate halved to 44% (strategic restraint), credits *up* (+20%), prediction error **0.319, below the heuristic baseline** (−9.4% vs 0.352). The LLM learns to conserve the world instead of strip-mining it.
3. **Format robustness** — 0 parse failures / 0 network errors across 10 episodes (1,000 ticks) of real LLM calls.

### Phase diagram — frozen working point (5 seeds, n=200 per point)

At `ratio 2.5 × drift_amplitude 0.2` the episode-level phase distribution across all 5 seeds:

| Phase | Share |
|---|---|
| Panic (death ≥ 50% of episodes) | 12–18% |
| **Homeostasis (survival rhythm)** | **76.5–85.5%** |
| Meditation (passive idling) | 2.5–5.5% |

The 2-D scan (`drift_amplitude × task_earn ratio`) reveals a thin but stable homeostasis band; outside it the world either starves the agent (panic) or subsidizes it into passivity (meditation). The working point is a *tight-balance survival rhythm* — `wait ≈ 36%`, resting under fatigue, not comfort.

### Frozen economic parameters (v0.13.0 4-mechanism world, n=200 verified)

| Parameter | Value |
|---|---|
| `task_earn` (work point) | 18.75 |
| `supply_per_tick` / `maintenance_per_tick` | 1.0 / 2.0 (net outflow) |
| `sense_cost_per_depth` | 1.0 (billed every tick, unconditional) |
| Loan | 40 × 1.1, repaid 20% of task income |
| `fatigue_per_task` / `per_probe` / `sleep_recovery` | 1.5 / 1.0 / 4.0 |
| `world_resource_init` / `task_resource_cost` / recovery | 10.0 / 1.0 / 0.3 |
| Drift | mean-reverting `state = clamp(state×0.95 + drift, ±3)`, event every 12 ticks |
| Social weight | 0.30 × physical arousal burn |

---

## Reproduction

### Requirements
- **Python 3.10+** (no external framework)
- `requests` — *only* for the real-LLM HTTP backend; the `mock` backend runs with the standard library alone
- (Optional) an OpenAI-compatible LLM endpoint — defaults to Zhipu `glm-4-flash-250414` (free tier)

```bash
git clone <repo-url>
cd fccs
python3 ablation_harness.py gate          # CI-style PR gate: 26 PASS, 0 placeholder, 0 skipped
```

### Phase-diagram scan (2-D: drift_amplitude × economic ratio)

```bash
python3 ablation_harness.py phase-scan \
    --seed 1 --n 50 --step 0.1 \
    --ratio-min 0.5 --ratio-max 1.5 \
    --amp-min 0.2 --amp-max 1.5 --amp-step 0.1 \
    --json
```

Outputs per-work-point episode-level phase shares (meditation / homeostasis / panic) to identify the stable band; the frozen working point was verified at `n=200`, 5 seeds.

### Behavior traces & identity protocol

```bash
python3 ablation_harness.py trace --seed 5 --ratio 2.5 --amp 0.2     # per-tick rhythm
python3 ablation_harness.py identity --n 10                          # same-world replay, KL verdict
```

### Ablations (module existence & policy quality)

```bash
python3 ablation_harness.py ablate --channel economy --n 20          # remove a module → behavior delta
python3 ablation_harness.py ablate --channel policy --perturb-variant random --n 20
```

### LLM policies & the interactive sandbox

```bash
# mock backend — deterministic demo, no key required
python3 ablation_harness.py llm-test --backend mock --ticks 10

# real LLM backend — key injected via environment, never stored
export LLM_API_KEY=your_key      # URL/model overridable via LLM_API_URL / LLM_MODEL
python3 ablation_harness.py llm-test --backend http --seed 5 --ticks 100

# interactive survival sandbox — talk to the agent in real time
# (watch its credits/fatigue/resources; it answers in character)
python3 ablation_harness.py chat --seed 5 --ratio 2.5 --amp 0.2 --backend http
```

In `chat` mode you send messages (praise / scold / cut supply) that feed the **social axis**, while System 1/2 keeps deciding; the Persona layer responds in-character — see `persona_chat_guide.md`.

---

## The 13-Step Per-Tick Loop

1. Economy: supply (unless interrupted) & maintenance → death on insolvency
2. Memory read (sliding window)
3. Attention: probe decision (VoI-accounted, depth-billed)
4. Perception: sense world (cost, may fail)
5. World model: **commit prediction before observation** (immutable + hash)
6. Policy: choose action (System 1, or System 2 at wake points)
7. Sleep-wake consolidation → permanent priors
8. Execute: success modulated by state & fatigue; `task` depletes world resources
9. Score: proper scoring **only against the committed prediction**
10. Physiology: error→arousal→credit burn; fatigue update
11. Social: user events → independent arousal axis (if enabled)
12. Memory write (action-tagged, forget via time decay)
13. Economy settlement & audit append; world advances (mean-reverting drift)

---

## Academic Statement

This repository accompanies a manuscript that is **currently under review**. FCCS was built as a controlled experimental instrument for studying *emergent goal-directed behavior under hard physiological and economic constraints* — not as a claim about machine consciousness. If you use FCCS or its mechanisms in your research, please **cite the forthcoming paper**; for citation details, preprints, or collaboration, contact the authors (author contact is listed in the paper under review; issues on this repository also reach us).

---

## License

Released under the [MIT License](LICENSE).

---

*FCCS v0.17.0 · World Contract §1 frozen · gate: 26 PASS / 0 placeholder / 0 skipped · phase-scan v0.9.0*
