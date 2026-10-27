### (a) T-60 test set, one row per model (test base rate 0.1795; threshold = max-F1 on September)
| Model | Train rows | PR-AUC (mean ± std over seeds) | Lift | ROC-AUC | F1 | Precision | Recall | Train s | Predict s | Seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost (native categoricals, tuned) | 3,836,893 | 0.6409 | 3.57x | 0.8214 | 0.561 | 0.600 | 0.527 | 196 | 6.6 | 1 |
| XGBoost (tuned, ordinal codes) | 3,836,893 | 0.6390 ± 0.0006 | 3.56x | 0.8191 ± 0.0005 | 0.560 | 0.594 | 0.530 | 84 | 1.7 | 3 |
| LightGBM (main) | 3,836,893 | 0.6387 ± 0.0001 | 3.56x | 0.8202 ± 0.0000 | 0.560 | 0.584 | 0.538 | 134 | 3.5† | 3 |
| XGBoost (native categoricals, default params) | 3,836,893 | 0.6364 | 3.54x | 0.8173 | 0.558 | 0.592 | 0.528 | 51 | 1.3 | 1 |
| XGBoost (default params, ordinal codes) | 3,836,893 | 0.6350 ± 0.0012 | 3.54x | 0.8165 ± 0.0010 | 0.556 | 0.597 | 0.521 | 115 | 0.8 | 3 |
| Random Forest | 1,000,000 | 0.6189 ± 0.0006 | 3.45x | 0.8128 ± 0.0001 | 0.550 | 0.593 | 0.512 | 36 | 2.0 | 3 |
| Decision Tree (CART) | 3,836,893 | 0.6041 ± 0.0000 | 3.36x | 0.7929 ± 0.0000 | 0.533 | 0.647 | 0.453 | 75 | 0.2 | 3 |
| Logistic Regression | 3,836,893 | 0.5502 | 3.06x | 0.7806 | 0.507 | 0.521 | 0.494 | 352 | 1.5 | 1 |
| KNN | 200,000 | 0.4784 ± 0.0029 | 2.66x | 0.7400 ± 0.0023 | 0.460 | 0.512 | 0.417 | 0 | 591.7 | 3 |
| Naive Bayes (Categorical) | 3,836,893 | 0.4593 | 2.56x | 0.7318 | 0.429 | 0.512 | 0.369 | 105 | 0.8 | 1 |
| Naive Bayes (Gaussian) | 3,836,893 | 0.3624 | 2.02x | 0.7198 | 0.436 | 0.340 | 0.607 | 7 | 0.6 | 1 |
| Majority dummy | 3,836,893 | 0.1795 | 1.00x | 0.5000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 1 |
† LightGBM model-only prediction time on 932,084 rows (3 runs); the cached training log also timed pandas slicing (62 s). Random Forest uses a 1M stratified sample, KNN a 200k stratified sample.

### (b) Matched training data
| training_data | model | PR-AUC / diff | Lift / verdict | PR-AUC 95% CI (flight-level) | roc_auc | train_s |
|---|---|---|---|---|---|---|
| 1M stratified sample | LightGBM on RF's 1M sample | 0.6333 ± 0.0004 | 3.53x | [0.6310, 0.6356] | 0.8158 | 16.1 |
| 1M stratified sample | XGBoost (tuned) on RF's 1M sample | 0.6376 ± 0.0006 | 3.55x | [0.6351, 0.6396] | 0.8185 | 38.0 |
| 1M stratified sample | Random Forest | 0.6189 ± 0.0006 | 3.45x | [0.6160, 0.6208] | 0.8128 | 35.6 |
| 1M stratified sample |   >> LightGBM minus Random Forest (paired) | +0.0149 | statistically real and practically meaningful | [0.0141, 0.0157] |  |  |
| 200k stratified sample | LightGBM on KNN's 200k sample | 0.6223 ± 0.0006 | 3.47x | [0.6200, 0.6246] | 0.8081 | 22.1 |
| 200k stratified sample | XGBoost (tuned) on KNN's 200k sample | 0.6299 ± 0.0007 | 3.51x | [0.6271, 0.6317] | 0.8133 | 27.3 |
| 200k stratified sample | KNN | 0.4784 ± 0.0029 | 2.66x | [0.4775, 0.4826] | 0.74 | 0.3 |
| 200k stratified sample |   >> LightGBM minus KNN (paired) | +0.1422 | statistically real and practically meaningful | [0.1405, 0.1439] |  |  |
| full training set (3.84M rows) | LightGBM (main) | 0.6387 ± 0.0001 | 3.56x | [0.6366, 0.6411] | 0.8202 | 134.2 |

### (c) LightGBM minus XGBoost, paired PR-AUC difference
| Comparison (T-60) | Mean diff | 95% CI, flight-level | 95% CI, day-clustered | Verdict (flight-level CI) |
|---|---|---|---|---|
| LightGBM minus XGBoost (tuned, ordinal codes) | +0.0004 | [-0.0002, +0.0011] | [-0.0019, +0.0031] | not statistically distinguishable (practically equivalent, CI inside +/-0.01) |
| LightGBM minus XGBoost (default params, ordinal codes) | +0.0031 | [+0.0025, +0.0037] | [+0.0007, +0.0053] | statistically real but practically negligible (< 0.01) |
| LightGBM minus XGBoost (native categoricals, tuned) | -0.0021 | [-0.0025, -0.0017] | [-0.0035, -0.0009] | statistically real but practically negligible (< 0.01) |
| LightGBM minus XGBoost (native categoricals, default params) | +0.0024 | [+0.0020, +0.0028] |  | statistically real but practically negligible (< 0.01) |

| Comparison (day-ahead) | Mean diff | 95% CI, flight-level | 95% CI, day-clustered |
|---|---|---|---|
| LightGBM minus XGBoost (tuned, ordinal codes) | +0.0111 | [+0.0101, +0.0121] | [+0.0043, +0.0181] |
| LightGBM minus XGBoost (default params, ordinal codes) | +0.0273 | [+0.0261, +0.0285] |  |
| LightGBM minus XGBoost (native categoricals, tuned) | -0.0037 | [-0.0045, -0.0029] | [-0.0083, +0.0011] |

### (d) LightGBM ablation (test; CIs flight-level bootstrap on seed 42)
| setting | n_features | seeds | pr_auc | pr_auc_ci | roc_auc | roc_auc_ci |
|---|---|---|---|---|---|---|
| Day-ahead, weather OFF | 21 | 1 | 0.2479 | [0.2463, 0.2494] | 0.6200 | [0.6186, 0.6214] |
| Day-ahead, weather ON | 37 | 3 | 0.3539 ± 0.0014 | [0.3500, 0.3544] | 0.6990 ± 0.0003 | [0.6973, 0.6999] |
| Day-ahead + schedule-known turn structure (weather ON) | 40 | 1 | 0.4174 | [0.4152, 0.4197] | 0.7242 | [0.7229, 0.7256] |
| T-60, weather OFF | 27 | 1 | 0.5983 | [0.5960, 0.6007] | 0.7805 | [0.7791, 0.7819] |
| T-60, weather ON  [main] | 43 | 3 | 0.6387 ± 0.0001 | [0.6366, 0.6411] | 0.8202 ± 0.0000 | [0.8190, 0.8214] |
| T-60 without month / week_of_year | 41 | 1 | 0.6433 | [0.6411, 0.6455] | 0.8233 | [0.8221, 0.8246] |
| T-60 without historical delay rates | 40 | 1 | 0.6386 | [0.6365, 0.6410] | 0.8203 | [0.8192, 0.8216] |

Paired effects:
| effect | delta_pr_auc | pr_auc_ci | delta_roc_auc | roc_auc_ci |
|---|---|---|---|---|
| Weather effect, day-ahead (ON minus OFF) | 0.1043 | [0.1023, 0.1062] | 0.0785 | [0.0772, 0.0798] |
| Weather effect, T-60 (ON minus OFF) | 0.0405 | [0.0395, 0.0414] | 0.0396 | [0.0388, 0.0405] |
| Propagation effect: T-60 minus day-ahead (weather ON) | 0.2866 | [0.2844, 0.2889] | 0.1216 | [0.1204, 0.1228] |
|   of which schedule-known turn structure | 0.0652 | [0.0637, 0.0667] | 0.0257 | [0.0250, 0.0263] |
|   of which live predecessor status / delays | 0.2214 | [0.2194, 0.2234] | 0.0959 | [0.0948, 0.0969] |
| Month / week_of_year effect (with minus without) | -0.0045 | [-0.0049, -0.0040] | -0.0031 | [-0.0035, -0.0028] |
| Historical-rate effect (with minus without) | 0.0001 | [-0.0001, 0.0004] | -0.0002 | [-0.0004, 0.0000] |

### (e) Top 10 SHAP features (mean |SHAP|, log-odds, 50k test flights)
| feature | mean_abs_shap |
|---|---|
| prev_leg_dep_delay | 0.505 |
| week_of_year | 0.245 |
| d_prcp | 0.173 |
| sched_dep_hour | 0.168 |
| route | 0.167 |
| prev_leg_status | 0.147 |
| origin | 0.139 |
| o_prcp | 0.138 |
| prev_leg_arr_delay | 0.123 |
| prev_leg_sched_arr_minus_t60 | 0.111 |

Gain by feature group:
| feature_group | gain_share_pct |
|---|---|
| Predecessor status/delay (T-60) | 42.27 |
| Airline / airport / route | 24.28 |
| Weather (origin+dest) | 11.69 |
| Schedule turn structure | 11.49 |
| Time of day / calendar | 9.11 |
| Historical delay rates | 1.01 |
| Distance / schedule | 0.15 |

### (f) Threshold transfer (val prevalence 13.0% -> test prevalence 17.95%)
| model | Sep-tuned threshold | Val F1 (Sep, at its best thr) | Test F1 @ Sep threshold | Best possible test F1 (diagnostic) | Test predicted-positive rate | F1 lost to transfer |
|---|---|---|---|---|---|---|
| XGBoost (native categoricals, tuned) | 0.5554 | 0.5093 | 0.5615 | 0.5616 | 0.1578 | 0.0002 |
| XGBoost (tuned, ordinal codes) | 0.5256 | 0.5071 | 0.5603 | 0.5605 | 0.1604 | 0.0002 |
| LightGBM (main) | 0.5630 | 0.5085 | 0.5602 | 0.5605 | 0.1656 | 0.0002 |
| XGBoost (native categoricals, default params) | 0.5808 | 0.5063 | 0.5583 | 0.5584 | 0.1600 | 0.0001 |
| XGBoost (default params, ordinal codes) | 0.5480 | 0.5015 | 0.5563 | 0.5565 | 0.1568 | 0.0002 |
| Random Forest | 0.5726 | 0.4953 | 0.5495 | 0.5500 | 0.1552 | 0.0004 |
| Decision Tree (CART) | 0.6636 | 0.4761 | 0.5327 | 0.5328 | 0.1255 | 0.0001 |
| Logistic Regression | 0.5506 | 0.4434 | 0.5071 | 0.5075 | 0.1702 | 0.0004 |
| KNN | 0.2600 | 0.4230 | 0.4598 | 0.4673 | 0.1463 | 0.0075 |
| Naive Bayes (Categorical) | 0.9372 | 0.4041 | 0.4287 | 0.4498 | 0.1294 | 0.0210 |
| Naive Bayes (Gaussian) | 0.0583 | 0.3671 | 0.4356 | 0.4437 | 0.3206 | 0.0081 |
