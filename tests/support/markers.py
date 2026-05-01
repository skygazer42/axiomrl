INTEGRATION_FILES = {
    "test_checkpoint_evaluate.py",
    "test_checkpoint_resume.py",
    "test_cli_config.py",
    "test_cli_workflows.py",
    "test_cli_zoo_leaderboard.py",
    "test_cli_zoo_report.py",
    "test_doctor_cli.py",
    "test_real_end_to_end_workflows.py",
    "test_runtime_evaluation_support_integration.py",
    "test_runtime_training_session_integration.py",
    "test_zoo_presets.py",
}

SMOKE_HINTS = ("smoke", "reference_script")
SLOW_FILES = {
    "test_real_end_to_end_workflows.py",
    "test_runtime_evaluation_support_integration.py",
    "test_runtime_training_session_integration.py",
    "test_zoo_presets.py",
}


def classify_test_path(path: str) -> set[str]:
    normalized = path.replace("\\", "/")
    filename = normalized.rsplit("/", maxsplit=1)[-1]
    markers: set[str] = set()

    if filename in INTEGRATION_FILES:
        markers.add("integration")
    elif any(hint in filename for hint in SMOKE_HINTS):
        markers.add("smoke")
    else:
        markers.add("unit")

    if filename in SLOW_FILES:
        markers.add("slow")

    return markers
