from axiomrl import zoo_cli
from axiomrl.zoo import app, core, leaderboard, manifests, reporting


def test_zoo_cli_reexports_internal_module_entrypoints() -> None:
    assert zoo_cli.resolve_manifest_source is manifests.resolve_manifest_source
    assert zoo_cli.load_manifest_with_source is manifests.load_manifest_with_source
    assert zoo_cli.apply_manifest_defaults_to_config_payload is manifests.apply_manifest_defaults_to_config_payload
    assert zoo_cli.build_report_payload is reporting.build_report_payload
    assert zoo_cli.build_leaderboard_payload is leaderboard.build_leaderboard_payload
    assert zoo_cli.build_parser is app.build_parser
    assert zoo_cli.main is app.main


def test_zoo_cli_reexports_leaderboard_choices() -> None:
    assert zoo_cli.LEADERBOARD_METRIC_CHOICES == leaderboard.LEADERBOARD_METRIC_CHOICES
    assert zoo_cli.COMPARE_TO_CHOICES == leaderboard.COMPARE_TO_CHOICES
    assert zoo_cli.SCORE_VIEW_CHOICES == leaderboard.SCORE_VIEW_CHOICES


def test_zoo_core_reexports_internal_module_entrypoints() -> None:
    assert core.resolve_manifest_source is manifests.resolve_manifest_source
    assert core.load_manifest_with_source is manifests.load_manifest_with_source
    assert core.apply_manifest_defaults_to_config_payload is manifests.apply_manifest_defaults_to_config_payload
    assert core.build_report_payload is reporting.build_report_payload
    assert core.build_leaderboard_payload is leaderboard.build_leaderboard_payload
    assert core.build_parser is app.build_parser
    assert core.main is app.main


def test_zoo_core_functions_are_defined_in_split_modules() -> None:
    assert core.resolve_manifest_source.__module__ == "axiomrl.zoo.manifests"
    assert core.build_report_payload.__module__ == "axiomrl.zoo.reporting"
    assert core.build_leaderboard_payload.__module__ == "axiomrl.zoo.leaderboard"
    assert core.build_parser.__module__ == "axiomrl.zoo.app"
    assert core.main.__module__ == "axiomrl.zoo.app"
