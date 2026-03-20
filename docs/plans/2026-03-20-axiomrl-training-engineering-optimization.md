# AxiomRL Training/Engineering Optimization Roadmap (Benchmarking Mature RL Packages)

> **Goal:** Use a consistent scorecard across mature RL libraries to identify high-leverage training/engineering improvements for AxiomRL, then turn those findings into an implementable backlog with acceptance criteria and 1–2 verified PoCs.

> **Primary target:** single-node training (1 GPU / a few GPUs), production-leaning users, CLI-first workflows.

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
| AxiomRL (baseline) | `ce59cf6` |  |  |  |  |  |  |  |  |  |  |  |
| SB3 |  |  |  |  |  |  |  |  |  |  |  |  |
| RL Baselines3 Zoo |  |  |  |  |  |  |  |  |  |  |  |  |
| CleanRL |  |  |  |  |  |  |  |  |  |  |  |  |
| Tianshou |  |  |  |  |  |  |  |  |  |  |  |  |
| Sample Factory |  |  |  |  |  |  |  |  |  |  |  |  |
| RLlib (rllib/) |  |  |  |  |  |  |  |  |  |  |  |  |
| TorchRL |  |  |  |  |  |  |  |  |  |  |  |  |
| d3rlpy |  |  |  |  |  |  |  |  |  |  |  |  |

---

## AxiomRL current baseline (facts only)

Key facts to anchor the scorecard:

- **Stable API tiering:** `rl_training` + `rl_training.core` (stable), `rl_training.experimental` (managed but faster moving), `rl_training.contrib` + zoo workflows (explicitly experimental). See `docs/compatibility.md`.
- **Entry points:** `axiomrl` uses `src/rl_training/cli.py` (`train/eval/resume/zoo`); `axiomrl-zoo` uses `src/rl_training/zoo_cli.py`.
- **Run structure:** created via `src/rl_training/experiment/runs.py` (run id includes algo/env/seed/timestamp); artifacts include `metadata.json`, `config.yaml`, `checkpoints/`, and `tensorboard/`.
- **Metadata contract:** `src/rl_training/runtime/run_utils.py` writes `metadata.json` and updates `latest_metrics` / `best_checkpoint` on checkpoint saves.
- **Zoo reporting:** reads `runs/*/metadata.json` and produces text/JSON/CSV reports via `src/rl_training/zoo/reporting.py`.

---

## Backlog (draft: will be updated after benchmarking)

### P0 (high leverage, low risk)

1. **Enrich `metadata.json` with environment + git context**
   - Acceptance: `metadata.json` records `created_at_utc`, command/argv, python/OS, and pinned versions for key deps; git commit + dirty flag when available; no failures when git is absent.
2. **Add `axiomrl doctor` for environment self-check**
   - Acceptance: `axiomrl doctor` returns exit code 0 and prints python/torch/cuda/gymnasium versions and key capabilities in a stable, greppable format.
3. **Document "dev/test dependencies" for pixel and render tests**
   - Acceptance: `pip install -e ".[dev]"` documents extra optional packages needed to run the full suite; tests that require them either skip cleanly or install guidance is explicit.

### P1 (bigger surface area)

1. **Config validation + schema documentation**
   - Candidate: Hydra/OmegaConf or a thin Pydantic validation layer over the YAML payload.
2. **CLI UX upgrade**
   - Candidate: Typer/Rich for better help text, tables, and progress output while preserving the existing argparse command surface.
3. **A first-class report command outside zoo**
   - Candidate: `axiomrl report --runs-dir ... --output json/csv` as a stable alias for common report usage.

### P2 (defer unless needed)

1. **Async sampler/learner split (Sample Factory direction)**
2. **Learner/module abstraction (RLlib direction)**

---

## PoC tracking

Record PoC outcomes here (measured time/cost, risk notes, follow-ups).

- PoC 1: metadata enrichment (TDD, tests updated)
- PoC 2: `axiomrl doctor` (TDD, new tests)

