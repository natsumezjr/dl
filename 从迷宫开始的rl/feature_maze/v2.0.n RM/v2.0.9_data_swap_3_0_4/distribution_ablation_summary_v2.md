# V2 Distribution Ablation Summary

Diagnostic only. Do not treat a single high success value as final RM conclusion.

| run_name | policy | selected_counts | pressure_status | bfs_tiebreak | pure_argmax | sample_bfs | sample_pure | handdraw_bfs | handdraw_pure |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| v2_rm_dist_hard_only_500_16ep | hard_only | {'hard': 500} | WARN_RED | 1.0 | 0.94 | 0.43 | 0.29 | 0.5 | 0.25 |
| v2_rm_dist_medium_hard_500_16ep | medium_hard | {'hard': 250, 'medium': 250} | WARN_RED | 1.0 | 0.98 | 0.43 | 0.29 | 0.5 | 0.25 |