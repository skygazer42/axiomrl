from rl_training.experiment.registry import get_algorithm_spec

REGISTERED_ALGORITHMS = (
    "a2c",
    "ars",
    "openai_es",
    "awr",
    "awac",
    "marwil",
    "bear",
    "bc",
    "decision_transformer",
    "impala",
    "appo",
    "bcq",
    "mopo",
    "pets",
    "cal_ql",
    "crossq",
    "crr",
    "edac",
    "curl",
    "rlpd",
    "rebrac",
    "xql",
    "drq",
    "drqv2",
    "agent57",
    "diamond",
    "horizon_imagination",
    "po_dreamer",
    "twisted",
    "eadream",
    "dreamerv3",
    "mow",
    "jowa",
    "efficientzero",
    "scalezero",
    "gumbel_muzero",
    "spr",
    "ppg",
    "her",
    "ppo",
    "trpo",
    "recurrent_ppo",
    "dqn",
    "iql",
    "sac",
    "cql",
    "discrete_sac",
    "td3_bc",
    "tqc",
    "redq",
    "td3",
)


def test_builtin_algorithm_specs_are_registered() -> None:
    for algorithm_name in REGISTERED_ALGORITHMS:
        spec = get_algorithm_spec(algorithm_name)

        assert spec.name == algorithm_name
        assert callable(spec.train_fn)
        assert callable(spec.evaluate_fn)
        assert callable(spec.predict_fn)
