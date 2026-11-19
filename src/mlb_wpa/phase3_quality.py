"""Quality gate for conditional metric calculation; never manufactures inputs."""


def evaluate_quality(official_games, reference_games, expected_games, official_pa,
                     validated_pa, represented_pa, aggregate_mismatches=0,
                     identity_mismatches=0):
    baseline = 4 / 715
    limit = max(2 * baseline, baseline + .01)
    reasons = []
    rate = None if not official_pa else (official_pa - validated_pa) / official_pa
    if official_games != expected_games:
        reasons.append("official_game_collection_incomplete")
    if reference_games != expected_games:
        reasons.append("reference_collection_incomplete")
    if represented_pa != official_pa:
        reasons.append("official_pa_representation_mismatch")
    if aggregate_mismatches:
        reasons.append("official_player_totals_disagree")
    if identity_mismatches:
        reasons.append("player_pa_identity_counts_disagree")
    if rate is None or rate > limit:
        reasons.append("missing_rate_materially_worse_than_pilot")
    return {"metrics_and_rankings_allowed": not reasons, "stop_reasons": reasons,
            "pilot_excluded_fraction": baseline, "material_worsening_limit": limit,
            "primary_excluded_fraction": rate}


def require_metrics_permission(quality):
    if not quality.get("metrics_and_rankings_allowed", False):
        raise RuntimeError("Metrics/rankings stopped by data-quality gate: " + ";".join(quality.get("stop_reasons", [])))
