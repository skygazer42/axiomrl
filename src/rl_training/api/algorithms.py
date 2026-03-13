from __future__ import annotations

from pathlib import Path
import shutil
from collections.abc import Sequence

from rl_training.experiment.checkpointing import load_checkpoint
from rl_training.experiment.config import TrainConfig
from rl_training.experiment.default_manager import DefaultExperimentManager
from rl_training.runtime.callbacks import Callback
from rl_training.runtime.trainer import TrainResult
from rl_training.runtime.types import MetricDict
from rl_training.runtime.workflows import evaluate_checkpoint, predict_checkpoint

_NO_CHECKPOINT_ERROR = "no checkpoint available; call learn() or load() first"


def _config_from_payload(payload: dict) -> TrainConfig:
    return TrainConfig(
        algo=payload["algo"],
        env_id=payload["env_id"],
        seed=int(payload["seed"]),
        total_timesteps=int(payload["total_timesteps"]),
        output_dir=Path(payload["output_dir"]),
        device=payload.get("device", "auto"),
        num_envs=int(payload.get("num_envs", 1)),
        eval_episodes=int(payload.get("eval_episodes", 5)),
        log_interval=int(payload.get("log_interval", 1)),
        checkpoint_interval=int(payload.get("checkpoint_interval", 1)),
        tags=tuple(payload.get("tags", ())),
        algo_kwargs=dict(payload.get("algo_kwargs", {})),
        env_kwargs=dict(payload.get("env_kwargs", {})),
    )


class ManagedAlgorithm:
    algo_name: str

    def __init__(
        self,
        config: TrainConfig,
        *,
        manager: DefaultExperimentManager | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        if config.algo != self.algo_name:
            raise ValueError(f"{type(self).__name__} requires config.algo={self.algo_name!r}, got {config.algo!r}")
        self.config = config
        self.manager = manager or DefaultExperimentManager()
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.last_result: TrainResult | None = None

    def learn(self, *, callbacks: Sequence[Callback] | None = None) -> TrainResult:
        if self.checkpoint_path is None:
            result = self.manager.setup(self.config, callbacks=callbacks).train()
        else:
            result = self.manager.resume(
                self.checkpoint_path,
                total_timesteps=self.config.total_timesteps,
                output_dir=self.config.output_dir,
                eval_episodes=self.config.eval_episodes,
                callbacks=callbacks,
            ).train()

        self.last_result = result
        self.checkpoint_path = result.checkpoint_path
        return result

    def evaluate(self, *, num_episodes: int | None = None) -> MetricDict:
        if self.checkpoint_path is None:
            raise ValueError(_NO_CHECKPOINT_ERROR)
        return evaluate_checkpoint(self.checkpoint_path, num_episodes=num_episodes)

    def predict(self, obs: object, *, deterministic: bool = True):
        if self.checkpoint_path is None:
            raise ValueError(_NO_CHECKPOINT_ERROR)
        return predict_checkpoint(self.checkpoint_path, obs, deterministic=deterministic)

    def save(self, path: str | Path) -> Path:
        if self.checkpoint_path is None:
            raise ValueError(_NO_CHECKPOINT_ERROR)

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.resolve() != self.checkpoint_path.resolve():
            shutil.copy2(self.checkpoint_path, destination)
        return destination

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        *,
        manager: DefaultExperimentManager | None = None,
    ) -> ManagedAlgorithm:
        resolved_path = Path(checkpoint_path)
        checkpoint_state = load_checkpoint(resolved_path)
        config = _config_from_payload(checkpoint_state.config)
        if config.algo != cls.algo_name:
            raise ValueError(f"{cls.__name__}.load expected algo={cls.algo_name!r}, got {config.algo!r}")
        return cls(config, manager=manager, checkpoint_path=resolved_path)


class PPO(ManagedAlgorithm):
    algo_name = "ppo"


class TRPO(ManagedAlgorithm):
    algo_name = "trpo"


class A2C(ManagedAlgorithm):
    algo_name = "a2c"


class ARS(ManagedAlgorithm):
    algo_name = "ars"


class OpenAIES(ManagedAlgorithm):
    algo_name = "openai_es"


class IMPALA(ManagedAlgorithm):
    algo_name = "impala"


class APPO(ManagedAlgorithm):
    algo_name = "appo"


class AWR(ManagedAlgorithm):
    algo_name = "awr"


class AWAC(ManagedAlgorithm):
    algo_name = "awac"


class MARWIL(ManagedAlgorithm):
    algo_name = "marwil"


class BC(ManagedAlgorithm):
    algo_name = "bc"


class DecisionTransformer(ManagedAlgorithm):
    algo_name = "decision_transformer"


class MOPO(ManagedAlgorithm):
    algo_name = "mopo"


class PETS(ManagedAlgorithm):
    algo_name = "pets"


class BCQ(ManagedAlgorithm):
    algo_name = "bcq"


class BEAR(ManagedAlgorithm):
    algo_name = "bear"


class CrossQ(ManagedAlgorithm):
    algo_name = "crossq"


class CRR(ManagedAlgorithm):
    algo_name = "crr"


class D4PG(ManagedAlgorithm):
    algo_name = "d4pg"


class DRQN(ManagedAlgorithm):
    algo_name = "drqn"


class R2D2(ManagedAlgorithm):
    algo_name = "r2d2"


class HER(ManagedAlgorithm):
    algo_name = "her"


class NAF(ManagedAlgorithm):
    algo_name = "naf"


class DDPG(ManagedAlgorithm):
    algo_name = "ddpg"


class EDAC(ManagedAlgorithm):
    algo_name = "edac"


class CURL(ManagedAlgorithm):
    algo_name = "curl"


class DrQ(ManagedAlgorithm):
    algo_name = "drq"


class DrQv2(ManagedAlgorithm):
    algo_name = "drqv2"


class DiscreteSAC(ManagedAlgorithm):
    algo_name = "discrete_sac"


class PPG(ManagedAlgorithm):
    algo_name = "ppg"


class DQN(ManagedAlgorithm):
    algo_name = "dqn"


class ExpectedSARSA(ManagedAlgorithm):
    algo_name = "expected_sarsa"


class ExpectedDoubleDQN(ManagedAlgorithm):
    algo_name = "expected_double_dqn"


class BoltzmannDQN(ManagedAlgorithm):
    algo_name = "boltzmann_dqn"


class BoltzmannDoubleDQN(ManagedAlgorithm):
    algo_name = "boltzmann_double_dqn"


class MellowmaxDQN(ManagedAlgorithm):
    algo_name = "mellowmax_dqn"


class SoftDQN(ManagedAlgorithm):
    algo_name = "soft_dqn"


class SoftDoubleDQN(ManagedAlgorithm):
    algo_name = "soft_double_dqn"


class AdvantageLearningDQN(ManagedAlgorithm):
    algo_name = "advantage_learning_dqn"


class PersistentAdvantageLearningDQN(ManagedAlgorithm):
    algo_name = "persistent_advantage_learning_dqn"


class MunchausenDQN(ManagedAlgorithm):
    algo_name = "munchausen_dqn"


class MunchausenDoubleDQN(ManagedAlgorithm):
    algo_name = "munchausen_double_dqn"


class CQLDQN(ManagedAlgorithm):
    algo_name = "cql_dqn"


class CQLDoubleDQN(ManagedAlgorithm):
    algo_name = "cql_double_dqn"


class ClippedDoubleDQN(ManagedAlgorithm):
    algo_name = "clipped_double_dqn"


class HystereticDQN(ManagedAlgorithm):
    algo_name = "hysteretic_dqn"


class C51DQN(ManagedAlgorithm):
    algo_name = "c51_dqn"


class CalQL(ManagedAlgorithm):
    algo_name = "cal_ql"


class CQL(ManagedAlgorithm):
    algo_name = "cql"


class NStepDQN(ManagedAlgorithm):
    algo_name = "n_step_dqn"


class QRDQN(ManagedAlgorithm):
    algo_name = "qr_dqn"


class IQN(ManagedAlgorithm):
    algo_name = "iqn"


class IQL(ManagedAlgorithm):
    algo_name = "iql"


class XQL(ManagedAlgorithm):
    algo_name = "xql"


class DoubleDQN(ManagedAlgorithm):
    algo_name = "double_dqn"


class DuelingDQN(ManagedAlgorithm):
    algo_name = "dueling_dqn"


class NoisyDQN(ManagedAlgorithm):
    algo_name = "noisy_dqn"


class PrioritizedDQN(ManagedAlgorithm):
    algo_name = "prioritized_dqn"


class RainbowDQN(ManagedAlgorithm):
    algo_name = "rainbow_dqn"


class SAC(ManagedAlgorithm):
    algo_name = "sac"


class TQC(ManagedAlgorithm):
    algo_name = "tqc"


class REDQ(ManagedAlgorithm):
    algo_name = "redq"


class RLPD(ManagedAlgorithm):
    algo_name = "rlpd"


class ReBRAC(ManagedAlgorithm):
    algo_name = "rebrac"


class TD3(ManagedAlgorithm):
    algo_name = "td3"


class TD3BC(ManagedAlgorithm):
    algo_name = "td3_bc"


class RecurrentPPO(ManagedAlgorithm):
    algo_name = "recurrent_ppo"
