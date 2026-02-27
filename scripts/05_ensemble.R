# ==============================================================================
# Script: 05_ensemble.R
# Author: Kok Shau Loon (2401978)
# Tasks: 
# 1. Random Forest (randomForest / ranger)
# 2. Advanced Boosting (XGBoost / LightGBM / CatBoost) - Up to you
# 3. Class Imbalance Handling (sampsize/weights + PR-AUC Optimization)
# 4. Evaluation (Confusion Matrix, PR-AUC, ROC)
# ==============================================================================

cat("\n--- Running 05_ensemble.R ---\n")

# 1. Load Required Libraries
library(randomForest) # For Random Forest (or 'ranger' for speed)
library(xgboost)      # For Extreme Gradient Boosting
library(pROC)         # For ROC curves
library(PRROC)        # For Precision-Recall Curves (crucial for imbalanced data)
library(caret)        # For Evaluation

# Ensure output directories exist
if(!dir.exists("outputs/figures/ensemble")) dir.create("outputs/figures/ensemble", recursive = TRUE)
if(!dir.exists("outputs/models")) dir.create("outputs/models", recursive = TRUE)

# 2. Load the Pre-processed Data
train_data <- readRDS("outputs/data/train_data.rds")
test_data  <- readRDS("outputs/data/test_data.rds")

# Prepare DMatrix for XGBoost (Requires numeric matrices)
X_train_mat <- model.matrix(Revenue ~ . - 1, data = train_data)
y_train_num <- ifelse(train_data$Revenue == "Yes", 1, 0)

X_test_mat <- model.matrix(Revenue ~ . - 1, data = test_data)
y_test_num <- ifelse(test_data$Revenue == "Yes", 1, 0)

cat("\n--- PART A: Random Forest (Bagging) ---\n")

# Calculate sample sizes for balanced Random Forest (Downsampling the majority class)
n_minority <- sum(train_data$Revenue == "Yes")
rf_sampsize <- c("No" = n_minority, "Yes" = n_minority)

# [TODO]: Train Random Forest model using randomForest(). Include 'sampsize = rf_sampsize'.
# rf_model <- randomForest(Revenue ~ ., data = train_data, ntree = 500, mtry = sqrt(ncol(train_data)-1), sampsize = rf_sampsize, importance = TRUE)

# [TODO]: Plot Variable Importance and save to outputs/figures/ensemble/
# png("outputs/figures/ensemble/rf_var_imp.png")
# varImpPlot(rf_model, main = "Random Forest Variable Importance")
# dev.off()

# [TODO]: Predict on train set, perform threshold tuning
# [TODO]: Predict on test set, apply threshold, generate caret::confusionMatrix


cat("\n--- PART B: Gradient Boosting (Unconstrained) ---\n")
cat("Note: You are encouraged to research and implement advanced tuning here!\n")

# Calculate scale_pos_weight for XGBoost
scale_weight <- sum(y_train_num == 0) / sum(y_train_num == 1)

# Base XGBoost setup (Feel free to completely rewrite this to use LightGBM, CatBoost, or advanced tuning grids/Bayesian Opt)
dtrain <- xgb.DMatrix(data = X_train_mat, label = y_train_num)
dtest  <- xgb.DMatrix(data = X_test_mat, label = y_test_num)

# [TODO]: Define advanced parameters. Optimize for "aucpr" (Area under PR curve)
# xgb_params <- list(
#   objective = "binary:logistic",
#   eval_metric = "aucpr",       # Focus on PR-AUC due to imbalance
#   scale_pos_weight = scale_weight,
#   eta = 0.05,
#   max_depth = 6,
#   subsample = 0.8,
#   colsample_bytree = 0.8
# )

# [TODO]: Perform Cross-Validation (xgb.cv) to find optimal early stopping rounds
# [TODO]: Train final xgb.train() model

# [TODO]: Predict probabilities, calculate PR-AUC using PRROC::pr.curve
# fg <- xgb_preds[y_test_num == 1]; bg <- xgb_preds[y_test_num == 0]
# pr_curve <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = TRUE)
# plot(pr_curve)

# [TODO]: Generate caret::confusionMatrix for the boosting model


# 4. Save Models for Comparison Phase (06_comparison.R)
# saveRDS(list(rf = rf_model, xgb = xgb_model), "outputs/models/ensemble_models.rds")

cat("\n--- Completed 05_ensemble.R ---\n")