# Data availability, provenance and source roles

All results concern the2025 MLB regular season. No source data were fetched during final assembly. Exact original URLs, parameters, retrieval times and SHA-256 receipts remain in reports/eows_2025_exploratory/application_provenance.json and original collection manifests. Final copied assets have their own source catalog and checksums.

## Local data and reproduction levels

The complete research workspace preserves the source cache, checkpoints, processed PA data, game boxes, model parameters and all historical results. The final portfolio is a compact presentation/audit collection, not a duplicate of the multi-stage raw cache. It contains real derived215-player tables, coverage/exclusion ledgers and frozen evidence summaries.

The GitHub ignore rules exclude large raw responses, checkpoints, temporary caches and unnecessary event shards. They retain the compact frozen independent-model PA Parquet, official player totals and per-game official batting JSON needed for fixed-data analytical reproduction. Raw integration tests and source-to-event reconstruction still require the original local cache. A public checkout without it must not claim full source-level reproduction. No automatic recollection or provider refresh is offered as a substitute.

## Sources and roles

MLB Stats API supplies recorded play/state/PA credits, boxes and season totals. Routes include /api/v1/schedule, /api/v1.1/game/{game_pk}/feed/live, /api/v1/game/{game_pk}/winProbability and /api/v1/stats. These are recorded-event/accounting sources; the portfolio's numeric season WPA/LI come from the separate frozen Markov model.

FanGraphs supplies the retained65-game numeric reference sample and established definitions, not full-season official statistics.4,800 source-covered PA yield4,765 verified LI comparisons and4,758 WPA comparisons. This is approximately2.60% numeric season coverage, not representative proof. No additional requests bypass rate limits or provider restrictions.

Baseball Savant and Retrosheet support historical definition/event checks. Savant terminal pitch deltas were not accepted as verified batter-PA WPA; Retrosheet narratives corroborate event order, not numeric WPA/LI. Neither silently fills missing season inputs.

Original source ownership and provider terms remain applicable. This portfolio does not grant rights to redistribute third-party source responses, imply provider endorsement or invent a project license. Publication should retain exact attribution and use the permitted access/distribution method. No credentials or account access are required to inspect the final artifacts.

## Denominator clarification

Some historical narrative summaries bundled model evaluation under the4,758 joint/WPA denominator. Frozen metric-specific results show LI n=4,765 and WPA n=4,758; the final report corrects that presentation without altering historical results. Source matching4,765/4,800, processing181,408/182,926 and numeric reference coverage~2.60% answer different questions.
