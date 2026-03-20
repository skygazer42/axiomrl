# AxiomRL Training/Engineering Optimization Roadmap (Benchmarking Mature RL Packages)

> **Goal:** Use a consistent scorecard across mature RL libraries to identify high-leverage training/engineering improvements for AxiomRL, then turn those findings into an implementable backlog with acceptance criteria and 1-2 verified PoCs.

> **Primary target:** single-node training (1 GPU / a few GPUs), production-leaning users, CLI-first workflows.

---

## Status (as of 2026-03-21)

- [x] PoC 1: metadata enrichment for run `metadata.json` (`created_at_utc`, command, versions, git context).
- [x] PoC 2: `axiomrl doctor` CLI environment self-check.
- [x] Dev/test deps: optional `.[dev]` extras updated and documented.
- [x] Run artifacts: `docs/run-artifacts.md` describes artifact layout and `metadata.json` schema.
- [x] Config schema + validation: `docs/config-schema.md` and stricter `axiomrl train` config errors.
- [x] First-class report + leaderboard commands: `axiomrl report` / `axiomrl leaderboard` aliases.
- [x] CLI startup hygiene: lazy exports + lazy imports to keep `axiomrl --version|doctor|report|leaderboard` fast.

---

## Why this document exists

AxiomRL already has a strong foundation (stable core API, YAML configs, `axiomrl train/eval/resume/zoo` workflows). The risk now is drifting into ad-hoc features without a consistent quality bar. This roadmap provides:

- A shared rubric for "engineering maturity" (configs, runs, logging, evaluation, resume, exports).
- A pinned set of reference projects to learn from (what to borrow vs what to avoid).
- A ranked backlog (P0/P1/P2) with testable acceptance criteria.

---

## Reference projects (pinned during benchmarking)

We benchmark a small set of repositories with different strengths:

- Stable-Baselines3 (SB3): stable algorithm core, save/load, callbacks, VecEnv conventions.
- RL Baselines3 Zoo: experiment orchestration patterns and run directory conventions.
- CleanRL: vertical-slice reference scripts, reproducibility discipline, benchmarking utilities.
- Tianshou: collector/buffer/trainer separation and modular interfaces.
- Sample Factory: runner/sampler/learner layering for performance-focused architecture.
- Ray RLlib (conceptual): configuration and module/learner separation patterns (not a direct template).
- TorchRL: PyTorch-native data collection, transforms/wrappers, replay buffers/collectors.
- d3rlpy: offline-RL oriented dataset/model management and evaluation patterns.

**Pinning rule:** every repo entry must record a commit hash + date of checkout in the scorecard.

---

## Scorecard rubric (0/1/2)

Each dimension is scored:

- **0 (missing):** capability absent or not practically usable.
- **1 (present):** works but is inconsistent, undocumented, or hard to operationalize.
- **2 (mature):** good defaults + documentation + tests or strong ecosystem adoption.

### Dimensions

1. **Config system**
   - Typed validation, defaults, overrides, composition/includes, and discoverability.
2. **CLI ergonomics**
   - Helpful errors, stable flags, `--help` quality, structured output, and good UX for common tasks.
3. **Run artifacts**
   - Directory structure, metadata, checkpoint naming/aliases, and portability.
4. **Reproducibility**
   - Seeds, env/version capture, command capture, and deterministic-mode guidance.
5. **Evaluation & reporting**
   - Eval cadence, aggregate statistics, exports (JSON/CSV), and regression detection hooks.
6. **Resume / save-load semantics**
   - Compatibility policy, config drift handling, and robust resume behavior.
7. **Observability**
   - TensorBoard/W&B integration, structured metrics, and profiling guidance.
8. **Interoperability & export**
   - Dataset/rollout export, model export, and "bring your own analysis" support.

---

## Scorecard template

Fill one row per repo.

| Repo | Pinned commit | Config | CLI | Artifacts | Repro | Eval | Resume | Obs | Export | Evidence (paths/links) | Borrowable patterns | Avoid copying |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|---|---|
| AxiomRL (baseline) | `ce59cf6` | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | `src/rl_training/cli.py`, `src/rl_training/runtime/run_utils.py`, `src/rl_training/zoo/reporting.py` | Zoo reporting + stable-core tiering | Pixel tests need extra deps |
| SB3 | `a72be40` (2026-03-18) | 1 | 0 | 2 | 1 | 1 | 2 | 1 | 1 | `stable_baselines3/common/base_class.py`, `common/callbacks.py`, `common/logger.py` | Stable base class + callback ecosystem | Not a CLI-first project |
| RL Baselines3 Zoo | `bb4bdf1` (2026-03-13) | 2 | 2 | 2 | 2 | 2 | 1 | 2 | 1 | `rl_zoo3/train.py`, `rl_zoo3/exp_manager.py` | Experiment manager layer | Large env-specific flag surface |
| CleanRL | `004f8a0` (2025-07-08) | 1 | 1 | 1 | 2 | 1 | 0 | 2 | 1 | `cleanrl/ppo.py`, `cleanrl_utils/benchmark.py` | Vertical-slice scripts + tyro CLI | Duplication is intentional |
| Tianshou | `1bbe05b` (2025-12-01) | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | `tianshou/data/collector.py`, `tianshou/trainer.py` | Collector/buffer/trainer boundaries | Broad abstraction surface |
| Sample Factory | `8b35494` (2026-01-29) | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | `sample_factory/launcher/run.py`, `sample_factory/cfg/arguments.py` | Runner isolation + operational hygiene | Complexity not needed for v1 |
| RLlib (rllib/) | `70e34a3` (2026-03-19) | 2 | 1 | 2 | 1 | 2 | 2 | 2 | 1 | `rllib/algorithms/algorithm_config.py`, `rllib/core/learner/learner.py` | Learner/module separation concepts | Too platform-like for AxiomRL v1 |
| TorchRL | `4e2e787` (2026-03-20) | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | `torchrl/collectors/collectors.py`, `torchrl/data/replay_buffers/` | Composable collectors + replay buffers | Hydra-heavy examples; pick selectively |
| d3rlpy | `38c34b6` (2025-09-11) | 1 | 1 | 2 | 1 | 2 | 2 | 1 | 2 | `d3rlpy/cli.py`, `d3rlpy/dataset/` | Offline-RL ergonomics + eval tooling | Different focus (offline-first) |

**Note:** scores above are an initial pass. As we extract more evidence, we should adjust scores and add “why” in the repo notes below.

---

## AxiomRL current baseline (facts only)

Key facts to anchor the scorecard:

- **Stable API tiering:** `rl_training` + `rl_training.core` (stable), `rl_training.experimental` (managed but faster moving), `rl_training.contrib` + zoo workflows (explicitly experimental). See `docs/compatibility.md`.
- **Entry points:** `axiomrl` uses `src/rl_training/cli.py` (`train/eval/resume/zoo`); `axiomrl-zoo` uses `src/rl_training/zoo_cli.py`.
- **Run structure:** created via `src/rl_training/experiment/runs.py` (run id includes algo/env/seed/timestamp); artifacts include `metadata.json`, `config.yaml`, `checkpoints/`, and `tensorboard/`.
- **Metadata contract:** `src/rl_training/runtime/run_utils.py` writes `metadata.json` and updates `latest_metrics` / `best_checkpoint` on checkpoint saves.
- **Zoo reporting:** reads `runs/*/metadata.json` and produces text/JSON/CSV reports via `src/rl_training/zoo/reporting.py`.

---

## Repo notes (evidence + patterns)

### Stable-Baselines3 (SB3)

Evidence (selected):

- Base algorithm and save/load: `references/stable-baselines3/stable_baselines3/common/base_class.py`
- Callbacks (eval/checkpoint/logging patterns): `references/stable-baselines3/stable_baselines3/common/callbacks.py`
- Logger interfaces: `references/stable-baselines3/stable_baselines3/common/logger.py`

Borrowable patterns:

- “Stable base class + extension via callbacks” keeps the core small while enabling operational concerns (eval, checkpointing, extra logging).

Avoid copying:

- SB3 is not designed as a CLI-first experiment runner; use it as an API maturity reference, not as a CLI template.

### RL Baselines3 Zoo

Evidence (selected):

- CLI entry and orchestration: `references/rl-baselines3-zoo/rl_zoo3/train.py`
- ExperimentManager pattern: `references/rl-baselines3-zoo/rl_zoo3/exp_manager.py`

Borrowable patterns:

- A dedicated experiment layer owning run dirs, evaluation cadence, and hyperparam sources keeps algorithm code clean.

Avoid copying:

- Environment-specific flag explosion; AxiomRL should keep a smaller stable CLI surface and push complexity into config presets.

### CleanRL

Evidence (selected):

- `tyro`-based typed CLI in scripts, e.g. `references/cleanrl/cleanrl/ppo.py`
- Benchmark helpers: `references/cleanrl/cleanrl_utils/benchmark.py`

Borrowable patterns:

- One readable “vertical slice” per algorithm is invaluable for debugging, regression hunting, and education (even if the library core is abstracted).

Avoid copying:

- Script duplication is intentional; for AxiomRL, keep vertical slices as references, not as the main architecture.

### Tianshou

Evidence (selected):

- Collector boundary: `references/tianshou/tianshou/data/collector.py`
- Trainer entry: `references/tianshou/tianshou/trainer.py`

Borrowable patterns:

- Strong separation between collection, buffer, and update logic makes it easier to test and extend.

Avoid copying:

- The abstraction surface is broad; borrow the boundaries, not the entire API model.

### Sample Factory

Evidence (selected):

- Launcher: `references/sample-factory/sample_factory/launcher/run.py`
- Argument/config plumbing: `references/sample-factory/sample_factory/cfg/arguments.py`

Borrowable patterns:

- Isolation of operational complexity into dedicated runner layers helps keep the rest of the codebase readable.

Avoid copying:

- Full async/perf machinery is overkill for AxiomRL’s single-node target; borrow patterns, not scale.

### Ray RLlib (rllib/)

Evidence (selected):

- Config object design: `references/ray/rllib/algorithms/algorithm_config.py`
- Learner concepts: `references/ray/rllib/core/learner/learner.py`

Borrowable patterns:

- Treat config as a buildable object and keep “module vs learner” separation in mind for future phases.

Avoid copying:

- RLlib is a distributed platform; avoid importing platform-level complexity into AxiomRL v1/v1.x.

### TorchRL

Evidence (selected):

- Collectors: `references/torchrl/torchrl/collectors/collectors.py`
- Replay buffers: `references/torchrl/torchrl/data/replay_buffers/`

Borrowable patterns:

- Composable collectors/buffers are excellent reference points for clean interfaces and testing.

Avoid copying:

- Hydra-driven examples and distributed layers should be adopted only where they match AxiomRL’s workflow.

### d3rlpy

Evidence (selected):

- Click-based tooling: `references/d3rlpy/d3rlpy/cli.py`

Borrowable patterns:

- Offline-first ergonomics: dataset handling, evaluation/report tooling, and model packaging ideas.

Avoid copying:

- The center of gravity is offline RL; adopt patterns where AxiomRL’s offline workflows match.

---

## Backlog (draft: will be updated after benchmarking)

### P0 (high leverage, low risk)

- [x] **Enrich `metadata.json` with environment + git context** (Done: 2026-03-21)
  - Acceptance (met): `metadata.json` records `created_at_utc`, command/argv, python/OS, and pinned versions for key deps; git commit + dirty flag when available; no failures when git is absent.
  - Implementation: `src/rl_training/runtime/run_utils.py`
  - Tests: `tests/test_run_utils.py`
  - Docs: `docs/run-artifacts.md` (schema + run dir layout)

- [x] **Add `axiomrl doctor` for environment self-check** (Done: 2026-03-21)
  - Acceptance (met): `axiomrl doctor` returns exit code 0 and prints python/torch/cuda/gymnasium versions and key capabilities in a stable, greppable format.
  - Implementation: `src/rl_training/cli.py`
  - Tests: `tests/test_doctor_cli.py`

- [x] **Document "dev/test dependencies" for pixel and render tests** (Done: 2026-03-21)
  - Acceptance (met): `pip install -e ".[dev]"` documents extra optional packages needed to run the full suite; tests that require them either skip cleanly or install guidance is explicit.
  - Packaging: `pyproject.toml` optional deps `dev` (includes `opencv-python`, `pygame`)
  - Docs: `README.md`
  - Tests: `tests/test_package_smoke.py`

### P1 (bigger surface area)

- [x] **Config validation + schema documentation** (Done: 2026-03-21)
  - CLI validation: `src/rl_training/cli.py`
  - Docs: `docs/config-schema.md`
  - Tests: `tests/test_cli.py`

- [ ] **CLI UX upgrade**
  - Candidate: Typer/Rich for better help text, tables, and progress output while preserving the existing argparse command surface.

- [x] **First-class report/leaderboard commands outside zoo** (Done: 2026-03-21)
  - Commands: `axiomrl report`, `axiomrl leaderboard` (aliases to `axiomrl zoo --format report|leaderboard`)
  - Docs: `README.md`
  - Tests: `tests/test_cli.py`

### P2 (defer unless needed)

1. **Async sampler/learner split (Sample Factory direction)**
2. **Learner/module abstraction (RLlib direction)**

---

## PoC tracking

Record PoC outcomes here (measured time/cost, risk notes, follow-ups).

- PoC 1: metadata enrichment (TDD; `src/rl_training/runtime/run_utils.py`, `tests/test_run_utils.py`)
- PoC 2: `axiomrl doctor` (TDD; `src/rl_training/cli.py`, `tests/test_doctor_cli.py`)
