#!/bin/bash
# Runs Step 4 end to end, one process per model family (avoids OpenMP runtime clashes, keeps timings clean).
cd "$(dirname "$0")"; PY=../.venv/bin/python; L=../results/model_comparison/logs; mkdir -p $L
set -x
$PY 04_train_compare.py --stages tune main --models lightgbm                 > $L/01_lgbm_tune_main.log 2>&1
$PY 04_train_compare.py --stages main --models xgboost                      > $L/02_xgb_main.log 2>&1
$PY 04_train_compare.py --stages main --models dummy_majority logistic_regression naive_bayes_gaussian naive_bayes_categorical decision_tree knn > $L/03_sk_main.log 2>&1
$PY 04_train_compare.py --stages main --models random_forest                > $L/04_rf_main.log 2>&1
$PY 04_train_compare.py --stages ablation --models lightgbm                 > $L/05_ablation_lgbm.log 2>&1
$PY 04_train_compare.py --stages ablation --models xgboost_native_cat       > $L/06_ablation_xgb_native.log 2>&1
$PY 04_train_compare.py --stages scalability --models lightgbm              > $L/07_scal_lgbm.log 2>&1
$PY 04_train_compare.py --stages scalability --models xgboost               > $L/08_scal_xgb.log 2>&1
$PY 04_train_compare.py --stages scalability --models logistic_regression   > $L/09_scal_lr.log 2>&1
$PY 04_train_compare.py --stages scalability --models random_forest         > $L/10_scal_rf.log 2>&1
$PY 04_train_compare.py --stages dayahead                                   > $L/11_dayahead_all.log 2>&1
$PY 04_train_compare.py --stages evaluate                                   > $L/12_evaluate.log 2>&1
echo ALL_DONE
