# Summary of Results: `05_ensemble.R`

------------------------------------------------------------------------

## Program Structure

```         
05_ensemble.R
├── Shared Helpers
│   ├── evaluate_rf()        — compute & print RF metrics from probability scores
│   ├── evaluate_xgb()       — compute & print XGBoost metrics from DMatrix predictions
│   ├── train_rf()           — train RF on train_data with optional parameters
│   └── get_oob_prauc()      — train RF, return PR-AUC + OOB error from OOB votes
│
├── PART A: Random Forest
│   ├── 1.  Default model training
│   ├── 2.  Tree stability analysis (stable ntree)
│   ├── 3.  Tune mtry
│   ├── 4.  Tune nodesize
│   ├── 5.  Tune sampsize
│   ├── 6.  Refinement (re-tune mtry with fixed sampsize)
│   ├── 7.  Train tuned model
│   ├── 8.  Default vs Tuned comparison (test set)
│   ├── 9.  Model selection + threshold optimisation
│   ├── 10. Feature importance
│   ├── 11. Partial dependence plots
│   ├── 12. Final evaluation
│   └── 13. Save final RF packet
│
└── PART B: XGBoost
    ├── 1.  Temporal data preparation
    ├── 2.  Default model training
    ├── 3.  Bayesian hyperparameter optimisation
    ├── 4.  Tuned model training
    ├── 5.  Default vs Tuned comparison (test set)
    ├── 6.  Threshold optimisation
    ├── 7.  Feature importance
    ├── 8.  Final evaluation
    └── 9.  Save final XGBoost packet
```

### Helper Function Reference

| Function | Purpose |
|----|----|
| `evaluate_rf(probs, actuals, threshold, label, verbose)` | Computes Precision, Recall, F1, Balanced Accuracy, PR-AUC from probability scores. `verbose=TRUE` prints the metric block; `verbose=FALSE` returns silently for use inside tuning loops. |
| `evaluate_xgb(model, dmatrix, actual_labels, threshold, label)` | Same metrics as `evaluate_rf` but takes a trained XGBoost model and DMatrix directly. Always prints. |
| `train_rf(ntree, mtry, nodesize, sampsize)` | Wrapper around `randomForest()` that builds the argument list dynamically — optional parameters are only passed if non-NULL, avoiding randomForest default override issues. |
| `get_oob_prauc(ntree, mtry, nodesize, sampsize)` | Trains one RF via `train_rf()` and returns a list of `prauc` (PR-AUC from OOB votes) and `oob_err` (final OOB error). Used as the tuning objective across all parameter steps. |
| `sens_row(res, selected_val, model_votes)` | Calls `evaluate_rf` internally and formats the output into a sensitivity table row, so that repeated pattern does not have to be written out after every tuning step. |

------------------------------------------------------------------------

# Part A: Random Forest (Bagging)

------------------------------------------------------------------------

## 1. Model Overview

Random Forest is a **bagging-based ensemble method** that:

-   trains many decision trees on different bootstrap samples
-   randomly selects features at each split
-   aggregates predictions through majority voting

This approach improves:

-   model stability
-   generalization performance
-   robustness to overfitting

------------------------------------------------------------------------

## 2. Model Comparison

Two Random Forest models were evaluated:

| Model | Description |
|----|----|
| Default Random Forest | Standard parameters (`ntree=500`, `mtry=sqrt(p)`) with 1:1 sampling ratio |
| Tuned Random Forest | Parameters optimised via OOB PR-AUC objective |

Both models were evaluated on the **test dataset**.

#### Performance Comparison (Before Threshold Tuning)

| Metric | `rf_model` (Default) | `rf_model_tune` (Tuned) | Winner |
|:---|:--:|:--:|:--:|
| **Balanced Accuracy** | **0.7905** | 0.7590 | `rf_model` |
| **Sensitivity (Recall)** | **0.6895** | 0.6045 | `rf_model` |
| **PR-AUC** | 0.6562 | **0.6579** | `rf_model_tune` (Marginal) |
| **Kappa** | **0.5598** | 0.5311 | `rf_model` |
| **Accuracy** | 0.8494 | **0.8492** | Negligible difference |

**Conclusion:** While the tuned model achieves a marginally higher PR-AUC (+0.002), it suffers a significant drop in **Sensitivity (−8.5pp)** and **Balanced Accuracy (−3.1pp)**. Since balanced accuracy is the primary operational metric and PR-AUC difference is negligible, the **Default Model** was selected as the final classifier.

The final model is further improved using **threshold tuning**. Results are shown at [Key Metrics Table].

------------------------------------------------------------------------

## 3. Pipeline Overview

```         
Train Default RF
      ↓
Find stable ntree (OOB error convergence)
      ↓
Tune mtry → nodesize → sampsize → Refinement
(all steps: OOB PR-AUC as objective, balanced sampling throughout)
      ↓
Train Tuned RF
      ↓
Compare Default vs Tuned on test set
      ↓
Select winner (balanced accuracy on test set)
      ↓
Optimise threshold (balanced accuracy on OOB votes)
      ↓
Final evaluation on Q4 test set
```

------------------------------------------------------------------------

## 4. Pipeline Implementation Summary

### Step 1 — Default Model Training

A baseline Random Forest model was trained with:

-   `ntree` = 500
-   `mtry` = `floor(sqrt(p))` = 5
-   balanced sampling: 1:1 (`sampsize = c(No = n_minority, Yes = n_minority)`)

### Step 2 — Tree Stability Analysis

The number of trees (`ntree`) was analysed using **Out-Of-Bag (OOB) error convergence**.

Procedure:

-   Train up to 800 trees with balanced sampling applied
-   Identify where OOB error stabilises: absolute difference between consecutive OOB errors stays below 0.0005 for 50 consecutive trees

Result: `stable_ntree = 437`

![rf_stability_plot](images/rf_stability_plot-01.png)

![The image is not pushed to GitHub, please run the code once (if you haven't) and check outputs/figures/ensemble/rf_var_imp.png](images/rf_stability_plot-01.png)

------------------------------------------------------------------------

### Step 3 — Hyperparameter Tuning

A **coordinate descent strategy** was applied, tuning one parameter at a time.

**Tuning objective: PR-AUC computed from OOB votes**

OOB votes are used instead of a separate holdout set (refer to [Step 1 — Temporal Data Preparation](#step-1-temporal-data-preparation)) because:

-   each tree only predicts on observations it was NOT trained on — leak-free by construction
-   OOB covers all quarters (Q1+Q2+Q3), making the signal more stable than any single holdout quarter
-   consistent with the `get_oob_prauc()` tuning function used throughout
-   OOB is a built-in characteristic of random forest

Parameters tuned:

| Parameter  | Purpose                                                       |
|------------|---------------------------------------------------------------|
| `mtry`     | Number of features randomly sampled at each split             |
| `nodesize` | Minimum number of samples in terminal nodes                   |
| `sampsize` | Controls majority:minority sampling ratio for class imbalance |

#### Sensitivity Table

| Parameter | Selected Value | OOB Error | PR-AUC (OOB) | Precision | Recall | F1 | Balanced Accuracy |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ntree | 437 | 0.1026 | 0.7569 | 0.5584 | 0.8873 | 0.6855 | 0.8940 |
| mtry | 14 | 0.1043 | 0.7760 | 0.5489 | 0.8981 | 0.6813 | 0.8968 |
| nodesize | 5 | 0.1067 | 0.7778 | 0.5457 | 0.9034 | 0.6804 | 0.8985 |
| sampsize | 1:2 | 0.0874 | 0.7792 | 0.6050 | 0.8283 | 0.6993 | 0.8759 |
| Refinement | 10 | 0.0873 | 0.7811 | 0.6074 | 0.8283 | 0.7009 | 0.8763 |

------------------------------------------------------------------------

### Step 4 — Parameter Refinement

After tuning `nodesize` and `sampsize`, `mtry` was re-evaluated with the final `sampsize` fixed. This ensures earlier tuning decisions remain valid under the complete parameter configuration.

------------------------------------------------------------------------

### Step 5 — Tuned Model Training

A new Random Forest was trained using the optimised parameters. Training performance (OOB) is compared with the default model.

| Metric            | Default (OOB) | Tuned (OOB) |
|:------------------|:-------------:|:-----------:|
| Precision         |    0.5530     |   0.6148    |
| Recall            |    0.8841     |   0.8305    |
| F1                |    0.6804     |   0.7065    |
| Balanced Accuracy |    0.8915     |   0.8784    |
| PR-AUC            |    0.7589     |   0.7788    |

------------------------------------------------------------------------

### Step 6 — Model Comparison (Test Set)

Both models evaluated on the test dataset at 0.5 threshold. Results shown at [2. Model Comparison].

------------------------------------------------------------------------

### Step 7 — Model Selection + Threshold Optimisation

**Model selection:** winner chosen by **balanced accuracy on the test set** (not PR-AUC). This is methodologically consistent — balanced accuracy is used for both model selection and threshold tuning.

**Threshold optimisation:** the optimal threshold is found by maximising balanced accuracy across OOB predicted probabilities of the winner model. OOB votes are used (not test set) to avoid leakage.

Result: `best_threshold = 0.31`

------------------------------------------------------------------------

### Step 8 — Feature Importance

Feature importance extracted using **Mean Decrease Gini (MDG)** as the primary metric. Mean Decrease Accuracy (MDA) is included as a supplementary reference.

Note: MDA is based on accuracy which is less appropriate for imbalanced datasets, so MDG is the primary criterion here.

![The image is not pushed to GitHub, please run the code once (if you haven't) and check outputs/figures/ensemble/rf_var_imp.png](images/rf_var_imp.png)

| Rank | Feature | Interpretation |
|:--:|:---|:---|
| 1 | `PageValues` | The model is able to separate **Window Shoppers** from **Serious Buyers**, meaning that users visiting high-value pages (e.g., checkout or product pages) show strong purchase intent. |
| 2 | `ExitRates` | Exit behaviour indicates engagement. The model is able to tell that low exit rates suggest strong interest, while high exit rates signal disengagement and low purchase likelihood. |
| 3 | `ProductRelated_Duration` | The model distinguishes longer time spent on product pages indicates deeper evaluation and stronger buying intention. |

------------------------------------------------------------------------

### Step 9 — Partial Dependence Analysis (PDP)

PDP plots generated for the top 3 features to show **how each feature influences purchase probability**, which feature importance alone does not reveal.

![The image is not pushed to GitHub, please run the code once (if you haven't) and check outputs/figures/ensemble/rf_pdp_plots.png](images/rf_pdp_plots.png)

**PageValues (Left):** Sharp increase in purchase probability once PageValues becomes positive. Users move from casual browsing to high purchase intent once they reach product-related pages.

Action: encourage users to reach product pages as early as possible.

**ExitRates (Middle):** Clear negative relationship. No matter how interested the customer was at the start, if the page is confusing or boring, they get bored very fast suggest higher exit behaviour significantly reduces the likelihood of purchase.

Action: pages with high ExitRates should be optimised to reduce friction.

**ProductRelated_Duration (Right):** Nonlinear relationship. Users who stay longer than the average time on product pages become significantly more likely to buy. (Notice the line shoots up around 0, which is the average time spent).

Action: extended browsing time is a strong conversion signal; targeted promotions may help close the sale.

------------------------------------------------------------------------

### Step 10 — Final Model Evaluation

**Selected model:** `rf_model` (Default Parameters) with `best_threshold = 0.31`

#### Confusion Matrix

|                    | **Actual: No** | **Actual: Yes** |
|:-------------------|:--------------:|:---------------:|
| **Predicted: No**  |   3,205 (TN)   |    267 (FN)     |
| **Predicted: Yes** |    507 (FP)    |    709 (TP)     |

#### Key Metrics Table

| Metric | Value | Interpretation |
|:---|:--:|:---|
| **Accuracy** | 83.49% | Overall correct predictions. |
| **Balanced Accuracy** | **79.49%** | Fairness metric averaging Sensitivity and Specificity. |
| **Sensitivity (Recall)** | **72.64%** | Correctly identifies 7 in 10 actual buyers. |
| **Specificity** | 86.34% | Correctly identifies non-buyers. |
| **Precision (PPV)** | 58.31% | Reliability when the model predicts "Yes". |
| **Kappa** | 0.5408 | Moderate agreement above random chance. |
| **PR-AUC** | **0.6562** | Primary model selection metric — threshold-independent. |

## ------------------------------------------------------------------------

# Part B: XGBoost (Extreme Gradient Boosting)

------------------------------------------------------------------------

## 1. Model Overview

XGBoost is a **gradient boosting ensemble method** that:

-   builds trees sequentially, each correcting errors of the previous
-   applies regularisation (L1/L2) to control overfitting
-   uses second-order gradient information for faster, more accurate updates

Key advantages over Random Forest for this task:

-   native support for class imbalance via `scale_pos_weight`
-   PR-AUC directly usable as the evaluation metric (`eval_metric = "aucpr"`)
-   more flexible regularisation through `gamma`, `lambda`, `alpha`

------------------------------------------------------------------------

## 2. Model Comparison

Two XGBoost models were evaluated:

| Model | Description |
|----|----|
| Default XGBoost | 30 trees, L1/L2 regularisation, `scale_pos_weight` applied |
| Tuned XGBoost | Bayesian-optimised hyperparameters, CV-selected nrounds |

Both models evaluated on the **test dataset**.

#### Performance Comparison (Before Threshold Tuning)

| Metric                   | Default |   Tuned    |       Winner       |
|:-------------------------|:-------:|:----------:|:------------------:|
| **Balanced Accuracy**    | 0.7698  | **0.7930** |       Tuned        |
| **Sensitivity (Recall)** | 0.6404  | **0.6977** |       Tuned        |
| **PR-AUC**               | 0.6462  | **0.6591** |       Tuned        |
| **Kappa**                | 0.5350  | **0.5605** |       Tuned        |
| **Precision**            | 0.6256  |   0.6214   | Default (Marginal) |

**Conclusion:** The tuned model outperforms the default across all primary metrics. The **Tuned Model** was selected as the final classifier.

------------------------------------------------------------------------

## 3. Pipeline Overview

```         
Temporal split: Q1+Q2 (tuning train) | Q3 (holdout) | Q4 (test)
      ↓
Train Default XGBoost (30 trees, scale_pos_weight)
      ↓
Bayesian Optimisation (15 rounds, PR-AUC on Q3 holdout)
      ↓
CV to find optimal nrounds on full Q1+Q2+Q3
      ↓
Train Tuned XGBoost
      ↓
Compare Default vs Tuned on test set
      ↓
Threshold optimisation (balanced accuracy on Q3 holdout)
      ↓
Final evaluation on Q4 test set
```

------------------------------------------------------------------------

## 4. Pipeline Implementation Summary

### Step 1 — Temporal Data Preparation {#step-1-temporal-data-preparation}

Unlike RF which uses OOB for internal evaluation, XGBoost requires an explicit holdout set for tuning:

| Split          |   Quarters   | Purpose                                 |
|:---------------|:------------:|:----------------------------------------|
| `dtrain_sub`   |   Q1 + Q2    | Bayesian optimisation training set      |
| `dholdout_sub` |      Q3      | Bayesian optimisation evaluation set    |
| `dtrain`       | Q1 + Q2 + Q3 | Final model training                    |
| `dtest`        |      Q4      | Final evaluation (untouched throughout) |

`scale_pos_weight = 7.07` — ratio of negative to positive class, applied to address the 4:1 imbalance.

------------------------------------------------------------------------

### Step 2 — Default Model Training

Baseline XGBoost trained with:

-   `nrounds` = 30
-   `lambda` = 10 (L2), `alpha` = 1 (L1)
-   `scale_pos_weight` = 7.07
-   `eval_metric` = `"aucpr"`

Training performance (on full Q1+Q2+Q3):

| Metric            | Value  |
|:------------------|:------:|
| Precision         | 0.6871 |
| Recall            | 0.9968 |
| F1                | 0.8135 |
| Balanced Accuracy | 0.9663 |
| PR-AUC            | 0.9384 |

Note: Near-perfect training scores indicate the default model with only 30 shallow trees has already memorised the training data. The test set results are the true measure of performance.

------------------------------------------------------------------------

### Step 3 — Bayesian Hyperparameter Optimisation

**Objective:** PR-AUC on Q3 holdout (trains on Q1+Q2 each iteration)

**Algorithm:** Gaussian Process with UCB (Upper Confidence Bound) acquisition and RBF (squared exponential) kernel

Search space:

| Parameter | Range | Purpose |
|:---|:--:|:---|
| `max_depth` | 2–4 | Tree depth — controls complexity |
| `eta` | 0.01–0.02 | Learning rate — shrinks each tree's contribution |
| `subsample` | 0.5–0.7 | Row sampling ratio per tree |
| `colsample_bytree` | 0.5–0.7 | Feature sampling ratio per tree |
| `gamma` | 10–20 | Minimum loss reduction required to split |
| `min_child_weight` | 15–30 | Minimum sum of instance weights in a leaf |

Best parameters found (Round 1, Value = 0.6127):

| Parameter          | Best Value |
|:-------------------|:----------:|
| `max_depth`        |     4      |
| `eta`              |   0.0152   |
| `subsample`        |   0.5915   |
| `colsample_bytree` |   0.6880   |
| `gamma`            |   19.04    |
| `min_child_weight` |     23     |

------------------------------------------------------------------------

### Step 4 — Tuned Model Training

Optimal `nrounds` determined via 5-fold cross-validation on `dtrain` (full Q1+Q2+Q3) with `early_stopping_rounds = 30`.

Final model trained on full `dtrain` at the CV-optimal `nrounds`.

Training performance (on full Q1+Q2+Q3):

| Metric            | Value  |
|:------------------|:------:|
| Precision         | 0.5396 |
| Recall            | 0.9206 |
| F1                | 0.6804 |
| Balanced Accuracy | 0.9047 |
| PR-AUC            | 0.8026 |

------------------------------------------------------------------------

### Step 5 — Model Comparison (Test Set)

Both models evaluated on test dataset at 0.5 threshold. Results shown at [2. Model Comparison].

------------------------------------------------------------------------

### Step 6 — Threshold Optimisation

Threshold optimised on **Q3 holdout** (consistent with the Bayesian tuning setup) by maximising balanced accuracy.

Result: `best_xgb_threshold = 0.39`

Note: Unlike RF where OOB votes are used for threshold tuning, XGBoost uses the Q3 holdout because XGBoost has no native OOB mechanism.

------------------------------------------------------------------------

### Step 7 — Feature Importance

Feature importance extracted using **Gain** It represents the **total or average reduction in prediction error** (loss) that a feature provides when it is used to split the data in the model's trees.

![The image is not pushed to GitHub, please run the code once (if you haven't) and check outputs/figures/ensemble/xgb_importance.png](images/xgb_importance.png)

| Rank | Feature | Interpretation |
|:--:|:---|:---|
| 1 | `PageValues` | Consistent with the Random Forest results, this remains the primary driver. The high Gain proves that transaction-linked page value is the most efficient filter for identifying real buyer. |
| 2 | `Quarter.Q3` | Serving as the holdout set for hyperparameter tuning, this feature became the model's primary reference for "high-intent" behavior. Its high Gain proves the model effectively captured the specific seasonal patterns (July–September) required to generalize to the year-end Q4 test set. |
| 3 | `ExitRates` | Consistent with Random Forest, high exit rates strongly reduce purchase likelihood. |

Note on Q4 test scope: The Q4 test set covers November and December only — October remains in the training set.

------------------------------------------------------------------------

### Step 8 — Final Model Evaluation

**Selected model:** Tuned XGBoost with `best_xgb_threshold = 0.39`

#### Confusion Matrix

|                    | **Actual: No** | **Actual: Yes** |
|:-------------------|:--------------:|:---------------:|
| **Predicted: No**  |   3,291 (TN)   |    288 (FN)     |
| **Predicted: Yes** |    421 (FP)    |    688 (TP)     |

#### Key Metrics Table

| Metric | Value | Interpretation |
|:---|:--:|:---|
| **Accuracy** | 84.88% | Percentage of total correct predictions. |
| **Balanced Accuracy** | **79.58%** | Fairness metric (averaging Sensitivity and Specificity). |
| **Sensitivity (Recall)** | **70.49%** | Ability to correctly identify actual buyers (Yes), 7 out of 10. |
| **Specificity** | 88.66% | Ability to correctly identify non-buyers (No). |
| **Precision (PPV)** | 62.04% | Reliability when the model predicts "Yes". |
| **Kappa** | 0.5632 | Moderate agreement above random chance. |
| **PR-AUC** | **0.6591** | Primary model selection metric — threshold-independent. |

------------------------------------------------------------------------

# Self-Comment on Training and Testing Gap

A train-test gap is present: OOB balanced accuracy is 89.2% vs test balanced accuracy of 79.5%, approximately a 10pp gap for **Random Forest**. Reason could be due to the temporal split (train on Q1+Q2+Q3, test on Q4) is a fundamentally harder evaluation than random splitting as the model is asked to generalise to a future quarter it has never seen. Second, Q4 may contain seasonal behaviour patterns (e.g. end-of-year purchasing) that are underrepresented in Q1+Q2+Q3. Third, the gap is consistent across both models (\~10-12pp), suggesting it could be a structural property of the temporal split. If the gap were to grow significantly on future data, retraining with more recent quarters would be the appropriate response.

For **XGBoost** the training balanced accuracy is 90.5% vs test balanced accuracy of 79.6%, approximately an 11pp gap, similar in to RF (\~10pp) for the same setup. The regularisation parameters (`lambda=10`, `alpha=1`, tight `gamma` and `min_child_weight` from Bayesian tuning) are doing their job, the tuned model's gap (11pp) is not larger than the default (19pp gap: 96.6% train vs 77.0% test), which confirms tuning reduced overfitting meaningfully.

------------------------------------------------------------------------

# Execution Time

Total execution time: **593.78 seconds (9.90 minutes)**

The majority of runtime is attributable to the RF parameter tuning loop (multiple `get_oob_prauc` calls across mtry, nodesize, sampsize, and refinement grids) and the XGBoost Bayesian optimisation (15 rounds × model training + CV).
