# V1 Distribution Ablation Summary

Diagnostic only. Success values are not final model conclusions.

| run_name | policy | selected_counts | rollout_success | shortest_path_success | avg_bfs_gap | sample_check | handdraw_check |
|---|---|---:|---:|---:|---:|---:|---:|
| v1_dist_balanced_e_m_h_5000_16ep | balanced_e_m_h | {'easy': 109, 'hard': 3169, 'medium': 1616} | 0.775 | 0.73 | 2.065 | 0.75 | 0.0 |
| v1_dist_hard_only_5000_16ep | hard_only | {'hard': 3169} | 0.835 | 0.83 | 2.315 | 0.75 | 0.0 |
| v1_dist_medium_hard_5000_16ep | medium_hard | {'hard': 3169, 'medium': 1616} | 0.725 | 0.69 | 2.83 | 0.75 | 0.0 |
| v1_dist_random_5000_16ep | random | {'easy': 109, 'hard': 3169, 'medium': 1616, 'out_of_range': 73} | 0.765 | 0.745 | 3.585 | 0.75 | 0.0 |