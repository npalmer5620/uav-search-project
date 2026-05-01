# SITL Grid vs RL Characterization

- Created UTC: 2026-04-30T17:41:17.869843+00:00
- Grid run: `artifacts/sitl_eval/compare_grid_90s_20260430T171908Z`
- RL run: `artifacts/sitl_eval/compare_rl_90s_20260430T173015Z`
- Duration: 90s SEARCH budget
- RL model: `artifacts/rl/search_policy_v2_candidate_actions_50k/model.zip`

| Metric | Grid | RL |
|---|---:|---:|
| Valid flight | True | True |
| Max altitude m | 4.21 | 4.23 |
| Search path m | 224 | 232 |
| Visible coverage | 98.0% | 96.0% |
| Targets detected | 6 | 6 |
| Targets reported matched | 4 | 3 |
| False reported locations | 3 | 8 |
| Time to first detection s | 1.16 | 7.24 |
| Time to first report s | 21.5 | 14.9 |
| Mean best detection error m | 0.389 | 0.276 |
| Mean best report error m | 1.94 | 1.43 |

Notes:
- Both runs passed the flight validator: reached SEARCH, climbed above 2m, and moved more than 5m in SEARCH.
- RL covered almost as much of the visible area and detected all in-bounds people, but produced fewer matched reports and more false reported locations than grid in this single run.
