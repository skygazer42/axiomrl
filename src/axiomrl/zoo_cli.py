from axiomrl.zoo.app import build_parser, main
from axiomrl.zoo.leaderboard import (
    COMPARE_TO_CHOICES,
    LEADERBOARD_METRIC_CHOICES,
    SCORE_VIEW_CHOICES,
    build_leaderboard_payload,
)
from axiomrl.zoo.manifests import (
    MANIFEST_DRIFT_TYPE_CHOICES,
    MANIFEST_DRIFT_TYPE_TO_SUMMARY_FIELD,
    apply_manifest_defaults_to_config_payload,
    load_manifest,
    load_manifest_with_source,
    resolve_manifest_source,
)
from axiomrl.zoo.reporting import build_report_payload

__all__ = [
    "COMPARE_TO_CHOICES",
    "LEADERBOARD_METRIC_CHOICES",
    "MANIFEST_DRIFT_TYPE_CHOICES",
    "MANIFEST_DRIFT_TYPE_TO_SUMMARY_FIELD",
    "SCORE_VIEW_CHOICES",
    "apply_manifest_defaults_to_config_payload",
    "build_leaderboard_payload",
    "build_parser",
    "build_report_payload",
    "load_manifest",
    "load_manifest_with_source",
    "main",
    "resolve_manifest_source",
]


if __name__ == "__main__":
    raise SystemExit(main())
