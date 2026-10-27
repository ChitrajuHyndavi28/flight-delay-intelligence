#!/bin/bash
# Second wave of Step 4 (after run_step4.sh): fair XGBoost tuning, matched-data fits, extra ablation, final evaluation, explainability.
cd "$(dirname "$0")"; PY=../.venv/bin/python; L=../results/model_comparison/logs; mkdir -p $L
until grep -q ALL_DONE $L/run_step4.out; do sleep 20; done
set -x
$PY 04_train_compare.py --stages xgb_tuned                               > $L/13_xgb_tuned.log 2>&1
$PY 04_train_compare.py --stages matched --models lightgbm               > $L/14_matched_lgbm.log 2>&1
$PY 04_train_compare.py --stages matched --models xgboost_tuned          > $L/15_matched_xgb.log 2>&1
$PY 04_train_compare.py --stages ablation2 --models lightgbm             > $L/16_ablation2.log 2>&1
$PY 04_train_compare.py --stages evaluate                                > $L/17_evaluate_final.log 2>&1
$PY 04_explain.py                                                        > $L/18_explain.log 2>&1
echo ALL_DONE_B
