# A summary of result for **Random Forest (Bagging) ensemble model**

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
| Tuned Random Forest | Parameters optimized using OOB error |

Both models were evaluated on the **test dataset**.

#### Performance Comparison (Before Threshold Tuning)

| Metric | `rf_model` (Default) | `rf_model_tune` | **Winner** |
|:---|:--:|:--:|:--:|
| **Balanced Accuracy** | **0.7910** | 0.7582 | `rf_model` |
| **Sensitivity (Recall)** | **0.6906** | 0.6004 | `rf_model` |
| **Kappa** | **0.5606** | 0.5320 | `rf_model` |
| **Accuracy** | 0.8496 | **0.8503** | `rf_model_tune` (Slightly) |

**Conclusion:** While the Tuned Model showed a marginally higher raw Accuracy, it suffered from a significant drop in **Sensitivity (-9%)**. The **Default Model** was selected as the final classifier due to its superior ability to identify the minority "Revenue = Yes" class.

The final model will further be improved using **threshold tuning**. The result are shown at [Key Metrics Table]

------------------------------------------------------------------------

## 3. Pipeline Overview

The pipeline follows these steps:

1.  Train Default Random Forest
2.  Identify Stable Number of Trees (`ntree`)
3.  Hyperparameter Tuning
    -   `mtry`
    -   `nodesize`
    -   `sampsize`
4.  Refinement of tuned parameters
5.  Train Tuned Random Forest
6.  Compare Default vs Tuned models
7.  Threshold Optimization
8.  Feature Importance Analysis
9.  Partial Dependence Plot (PDP) Analysis
10. Final Model Evaluation

------------------------------------------------------------------------

## 4. Pipeline Implementation Summary

### Step 1 — Default Model Training

A baseline Random Forest model was trained with:

-   ntree = 500
-   mtry = sqrt(number of predictors), 5 in this case
-   balanced sampling (1:1)

### Step 2 — Tree Stability Analysis

The number of trees (`ntree`) was analyzed using **Out-of-Bag (OOB) error convergence**.

Procedure:

-   Train up to **800 trees**
-   Identify where OOB error stabilizes by evaluating the difference between OOB error for 50 continuuos trees is less than 0.005

Result:

ntree = 564

![The image is not pushed to GitHub, please run the code once (if you haven't) and check outputs/figures/ensemble/rf_stability_plot.png](images/rf_stability_plot-01.png)

------------------------------------------------------------------------

### Step 3 — Hyperparameter Tuning

A **coordinate descent tuning strategy** was applied.

Parameters tuned:

| Parameter  | Purpose                                  |
|------------|------------------------------------------|
| `mtry`     | Number of features sampled at each split |
| `nodesize` | Minimum samples in terminal nodes        |
| `sampsize` | Controls class imbalance during sampling |

Selection criterion:

-   Minimum OOB error

A sensitivity table was generated to track performance changes.

Note: The error after getting the `sampsize` increased is expected since the data now become less imbalanced.

| Parameter | Selected_Value | Best_OOB_Error | Precision | Recall | F1 | Balanced_Accuracy |
|----|----|----|----|----|----|----|
| ntree | 564 | 0.0746 | 0.7631 | 0.5773 | 0.6573 | 0.7759 |
| mtry | 14 | 0.0721 | 0.7413 | 0.6395 | 0.6866 | 0.8039 |
| nodesize | 5 | 0.0730 | 0.7364 | 0.6266 | 0.6771 | 0.7974 |
| sampsize | 1:2 | 0.0877 | 0.6055 | 0.8315 | 0.7007 | 0.8774 |
| Refinement | 4 | 0.0834 | 0.6180 | 0.8122 | 0.7019 | 0.8706 |

------------------------------------------------------------------------

### Step 4 — Parameter Refinement

After tuning `nodesize` and `sampsize`, `mtry` was re-evaluated to confirm optimal configuration.

This ensures earlier tuning decisions remain valid under the final setup.

------------------------------------------------------------------------

### Step 5 — Tuned Model Training

A new Random Forest model was trained using the tuned parameters.

Training metrics were compared with the default model.

------------------------------------------------------------------------

### Step 6 — Model Comparison (Test Data)

Both models were evaluated using the **test dataset confusion matrix**.

Metrics compared:

-   Precision
-   Recall
-   F1 Score
-   Balanced Accuracy

Result:

Shown at Section [2. Model Comparison]

------------------------------------------------------------------------

### Step 7 — Threshold Optimization

The classification threshold was tuned using **OOB predicted probabilities** to optimize **balanced accuracy**.

The optimized threshold was applied to the **test dataset for final evaluation**.

------------------------------------------------------------------------

### Step 8 - Feature Importance

Feature importance was extracted using **Mean Decrease Gini (MDG)**.

Note:

The **Mean Decrease Accuracy (MDA)** will only be used as an auxiliary metric to support MDG and for additional information as it uses accuracy which is not very appropriate to handle imbalanced dataset.

![The image is not pushed to GitHub, please run the code once (if you haven't) and check outputs/figures/ensemble/rf_var_imp.png](images/rf_var_imp.png)

| Rank | Feature | Interpretation |
|----|----|----|
| 1 | `PageValues` | The model is able to separate **Window Shoppers** from **Serious Buyers**, meaning that users visiting high-value pages (e.g., checkout or product pages) show strong purchase intent. |
| 2 | `ExitRates` | Exit behaviour indicates engagement. The model is able to tell that low exit rates suggest strong interest, while high exit rates signal disengagement and low purchase likelihood. |
| 3 | `ProductRelated_Duration` | The model distinguishes longer time spent on product pages indicates deeper evaluation and stronger buying intention. |

------------------------------------------------------------------------

### Step 9 - Partial Dependence Analysis (PDP)

Feature importance ranks variables but does not show **how they influence predictions**.

PDP plots were generated for the **top 3 features**.

![The image is not pushed to GitHub, please run the code once (if you haven't) and check outputs/figures/ensemble/rf_pdp_plots.png](images/rf_pdp_plots.png)

------------------------------------------------------------------------

### PageValues (Left)

The PDP shows a **sharp increase in purchase probability** once PageValues becomes positive.

Insight:

Users move from **casual browsing to high purchase intent** once they reach product-related pages.

Action:

Encourage users to reach product pages as soon as possible.

------------------------------------------------------------------------

### ExitRates (Middle)

The PDP shows a **negative relationship** with purchase probability.

Insight:

No matter how interested the customer was at the start, if the page is confusing or boring, they get Cold very fast suggest higher exit behaviour significantly reduces the likelihood of purchase.

Action:

Pages with high ExitRates should be optimized to reduce friction.

------------------------------------------------------------------------

### ProductRelated_Duration (Right)

The PDP shows a **nonlinear relationship**.

Insight:

Users who stay longer than the average time on product pages become significantly more likely to buy. (Notice the line shoots up around 0 (which is the average time spent).

Action:

When users spend extended time on product pages, targeted promotions may help convert them.

------------------------------------------------------------------------

### Step 10 - Conclusion

The Random Forest ensemble model successfully identifies **user engagement patterns that predict purchase behaviour**.

Key predictors include:

-   visiting high-value pages (`PageValues`)
-   sustained browsing (`ProductRelated_Duration`)
-   low exit behaviour (`ExitRates`)

The final selected model is:

-   `rf_model` (Default Parameters) + `best_threshold`

#### **Confusion Matrix**

|                    | **Actual: No** | **Actual: Yes** |
|:-------------------|:--------------:|:---------------:|
| **Predicted: No**  |   3,154 (TN)   |    258 (FN)     |
| **Predicted: Yes** |    558 (FP)    |  **718 (TP)**   |

#### **Key Metrics Table**

| Metric | Value | Interpretation |
|:---|:---|:---|
| **Accuracy** | 82.59% | Percentage of total correct predictions. |
| **Balanced Accuracy** | **79.27%** | Fairness metric (averages Sensitivity and Specificity). |
| **Sensitivity (Recall)** | **73.57%** | Ability to correctly identify actual buyers (Yes). |
| **Specificity** | 84.97% | Ability to correctly identify non-buyers (No). |
| **Kappa** | 0.5258 | Agreement level above random chance (Moderate). |
| **Precision (PPV)** | 56.27% | Reliability when the model predicts a "Yes". |

------------------------------------------------------------------------

### 

(remember to add the complete code structure)
