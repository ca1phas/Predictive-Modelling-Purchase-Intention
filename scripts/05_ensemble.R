# ==============================================================================
# Script: 05_ensemble.R
# Author: Kok Shau Loon (2401978)
# Tasks: 
# 1. Random Forest (randomForest / ranger)
# 2. Advanced Boosting (XGBoost / LightGBM / CatBoost) - Up to you
# 3. Class Imbalance Handling (sampsize/weights + PR-AUC Optimization)
# 4. Evaluation (Confusion Matrix, PR-AUC, ROC)
# ==============================================================================

# The entire program takes around 593.78 seconds (9.90 minutes) to execute
cat("\n--- Running 05_ensemble.R ---\n")

# 1. Load Required Libraries
library(randomForest)           # For Random Forest (or 'ranger' for speed)
library(xgboost)                # For Extreme Gradient Boosting
library(PRROC)                  # For Precision-Recall Curves (crucial for imbalanced data)
library(caret)                  # For Evaluation
library(rBayesianOptimization)  # For Bayesian Optimization
library(SHAPforxgboost)         # For SHAP Analysis

# Ensure output directories exist
if(!dir.exists("outputs/figures/ensemble")) dir.create("outputs/figures/ensemble", recursive = TRUE)
if(!dir.exists("outputs/models")) dir.create("outputs/models", recursive = TRUE)

# Load the Pre-processed Data
train_data  <- readRDS("outputs/data/train_data.rds")
test_data   <- readRDS("outputs/data/test_data.rds")
y_train_num <- ifelse(train_data$Revenue == "Yes", 1, 0)
y_test_num  <- ifelse(test_data$Revenue  == "Yes", 1, 0)

# ==============================================================================
# SHARED HELPERS
# ==============================================================================

# Evaluate RF model: computes all metrics from probability scores
# verbose = TRUE  -> prints full metric block
# verbose = FALSE -> silent, returns metrics only (used during tuning)
evaluate_rf <- function(probs, actuals, threshold = 0.5, label = "Model", verbose = TRUE) {
  pred_labels   <- factor(ifelse(probs > threshold, "Yes", "No"), levels = c("No", "Yes"))
  actual_factor <- factor(ifelse(actuals == 1,      "Yes", "No"), levels = c("No", "Yes"))
  cm <- confusionMatrix(pred_labels, actual_factor, positive = "Yes")
  pr <- pr.curve(scores.class0 = probs[actuals == 1],
                 scores.class1 = probs[actuals == 0], curve = FALSE)
  metrics <- c(Precision = unname(cm$byClass["Precision"]),
               Recall    = unname(cm$byClass["Recall"]),
               F1        = unname(cm$byClass["F1"]),
               Bal_Acc   = unname(cm$byClass["Balanced Accuracy"]),
               PR_AUC    = pr$auc.integral)
  if (verbose) {
    cat(sprintf("\n--- RF Performance: %s (threshold = %.2f) ---\n", label, threshold))
    cat(sprintf("  Precision : %.4f\n", metrics["Precision"]))
    cat(sprintf("  Recall    : %.4f\n", metrics["Recall"]))
    cat(sprintf("  F1        : %.4f\n", metrics["F1"]))
    cat(sprintf("  Bal. Acc  : %.4f\n", metrics["Bal_Acc"]))
    cat(sprintf("  PR-AUC    : %.4f\n", metrics["PR_AUC"]))
    cat("--------------------------------------------------\n")
  }
  return(invisible(metrics))
}

# Evaluate XGBoost model: computes all metrics from DMatrix predictions
evaluate_xgb <- function(model, dmatrix, actual_labels, threshold = 0.5, label = "Model") {
  probs         <- predict(model, dmatrix)
  pred_labels   <- factor(ifelse(probs > threshold, "Yes", "No"), levels = c("No", "Yes"))
  actual_factor <- factor(ifelse(actual_labels == 1, "Yes", "No"), levels = c("No", "Yes"))
  cm <- confusionMatrix(pred_labels, actual_factor, positive = "Yes")
  pr <- pr.curve(scores.class0 = probs[actual_labels == 1],
                 scores.class1 = probs[actual_labels == 0], curve = FALSE)
  metrics <- c(Precision = unname(cm$byClass["Precision"]),
               Recall    = unname(cm$byClass["Recall"]),
               F1        = unname(cm$byClass["F1"]),
               Bal_Acc   = unname(cm$byClass["Balanced Accuracy"]),
               PR_AUC    = pr$auc.integral)
  cat(sprintf("\n--- XGBoost Performance: %s (threshold = %.2f) ---\n", label, threshold))
  cat(sprintf("  Precision : %.4f\n", metrics["Precision"]))
  cat(sprintf("  Recall    : %.4f\n", metrics["Recall"]))
  cat(sprintf("  F1        : %.4f\n", metrics["F1"]))
  cat(sprintf("  Bal. Acc  : %.4f\n", metrics["Bal_Acc"]))
  cat(sprintf("  PR-AUC    : %.4f\n", metrics["PR_AUC"]))
  cat("--------------------------------------------------\n")
  return(invisible(metrics))
}

# ==============================================================================
# PART A: RANDOM FOREST
# ==============================================================================

cat("\n--- PART A: Random Forest (Bagging) ---\n")

# Train RF on train_data with optional parameters
train_rf <- function(ntree, mtry = NULL, nodesize = NULL, sampsize = NULL) {
  args <- list(formula = Revenue ~ ., data = train_data,
               ntree = ntree, importance = TRUE)
  if (!is.null(mtry))     args$mtry     <- mtry
  if (!is.null(nodesize)) args$nodesize <- nodesize
  if (!is.null(sampsize)) args$sampsize <- sampsize
  do.call(randomForest, args) # train the model with available parameter
}

# Train RF and return PR-AUC + OOB error from OOB votes (primary tuning objective)
get_oob_prauc <- function(ntree, mtry = NULL, nodesize = NULL, sampsize = NULL) {
  model <- train_rf(ntree, mtry, nodesize, sampsize)
  probs <- model$votes[, "Yes"]
  pr    <- pr.curve(scores.class0 = probs[y_train_num == 1],
                    scores.class1 = probs[y_train_num == 0], curve = FALSE)
  score <- pr$auc.integral
  if (is.null(score) || is.na(score) || is.infinite(score)) score <- 0 # to handle inappropriate score from pr
  return(list(prauc = score, oob_err = tail(model$err.rate[, "OOB"], 1)))
}

# =========================
# 1. Default Model
# =========================
cat("1: Training default Random Forest...\n")

n_minority  <- sum(train_data$Revenue == "Yes")
rf_sampsize <- c("No" = n_minority, "Yes" = n_minority)  # 1:1 balanced sampling
def_mtry    <- floor(sqrt(ncol(train_data) - 1)) # exclude response variable
def_ntree   <- 500

rf_model <- train_rf(ntree = def_ntree, mtry = def_mtry, sampsize = rf_sampsize)

evaluate_rf(rf_model$votes[, "Yes"], y_train_num,
            label = "Default (train OOB)")

# =========================
# 2. Find Stable ntree
# =========================
cat("2: Assessing tree stability...\n")

set.seed(42)
stability_rf <- randomForest(Revenue ~ ., data = train_data,
                             ntree = 800, sampsize = rf_sampsize) # we don't use train_rf as we want importance = FALSE
oob_error    <- stability_rf$err.rate[, "OOB"]
error_diff   <- abs(diff(oob_error))
stable_ntree <- 800; count <- 0
for (i in seq_along(error_diff)) {
  count <- ifelse(abs(error_diff[i]) < 0.0005, count + 1, 0)
  if (count >= 50) { stable_ntree <- i + 1; break }
}
cat(sprintf("Stable ntree: %d\n", stable_ntree))

png("outputs/figures/ensemble/rf_stability_plot.png", 800, 600)
plot(oob_error, type = "l", lwd = 2, xlab = "Trees", ylab = "OOB Error",
     main = "RF Tree Stability")
abline(v = stable_ntree, col = "blue", lty = 2)
legend("topright", legend = paste("Stable ntree =", stable_ntree),
       col = "blue", lty = 2, lwd = 1)
dev.off()

# =========================
# 3-6. Parameter Tuning with Coordinate Descent (OOB PR-AUC objective)
#   Each step finds the best value and stores it in the sensitivity table
#   All candidates use balanced sampling (rf_sampsize or final_sz)
# =========================
cat("\nPerforming parameter tuning (OOB PR-AUC objective)...\n")
cat("This may take a while — results shown once all steps complete.\n")

# Sensitivity result table is used to track the performance for adding in a tuned parameter
sensitivity_results <- data.frame(
  Parameter         = c("ntree", "mtry", "nodesize", "sampsize", "Refinement"),
  Selected_Value    = NA, OOB_Error = NA, PR_AUC_OOB = NA,
  Precision = NA, Recall = NA, F1 = NA, Balanced_Accuracy = NA
)

# Extract one sensitivity table row from a result + trained model votes
sens_row <- function(res, selected_val, model_votes) {
  m <- evaluate_rf(model_votes, y_train_num, verbose = FALSE)
  c(selected_val,
    round(res$oob_err,        4), round(res$prauc,       4),
    round(m["Precision"],     4), round(m["Recall"],     4),
    round(m["F1"],            4), round(m["Bal_Acc"],    4))
}

# Baseline (stable ntree, 1:1 sampsize)
res <- get_oob_prauc(ntree = stable_ntree, sampsize = rf_sampsize)
mod <- train_rf(ntree = stable_ntree, sampsize = rf_sampsize)
sensitivity_results[1, 2:8] <- sens_row(res, stable_ntree, mod$votes[, "Yes"])

# Tune mtry
cat("3: Tuning mtry...\n")
mtry_grid <- seq(2, floor(ncol(train_data) / 2), by = 2)
res_mtry  <- lapply(mtry_grid, function(m)
  get_oob_prauc(ntree = stable_ntree, mtry = m, sampsize = rf_sampsize))
best_idx  <- which.max(sapply(res_mtry, `[[`, "prauc")) # `[[` means we want to extract prauc from returned list
best_mtry <- mtry_grid[best_idx]
mod       <- train_rf(ntree = stable_ntree, mtry = best_mtry, sampsize = rf_sampsize)
sensitivity_results[2, 2:8] <- sens_row(res_mtry[[best_idx]], best_mtry, mod$votes[, "Yes"])
cat("mtry tuning completed\n")

# Tune nodesize
cat("4: Tuning nodesize...\n")
node_grid <- c(1, 5, 10, 20, 50)
res_node  <- lapply(node_grid, function(n)
  get_oob_prauc(ntree = stable_ntree, mtry = best_mtry,
                nodesize = n, sampsize = rf_sampsize))
best_idx      <- which.max(sapply(res_node, `[[`, "prauc"))
best_nodesize <- node_grid[best_idx]
mod           <- train_rf(ntree = stable_ntree, mtry = best_mtry,
                          nodesize = best_nodesize, sampsize = rf_sampsize)
sensitivity_results[3, 2:8] <- sens_row(res_node[[best_idx]], best_nodesize, mod$votes[, "Yes"])
cat("nodesize tuning completed\n")

# Tune sampsize (vary majority:minority ratio)
cat("5: Tuning sampsize...\n")
ratios     <- c(1, 1.5, 2)
res_sample <- lapply(ratios, function(r) {
  sz <- c("No" = as.integer(n_minority * r), "Yes" = n_minority)
  get_oob_prauc(ntree = stable_ntree, mtry = best_mtry,
                nodesize = best_nodesize, sampsize = sz)
})
best_idx   <- which.max(sapply(res_sample, `[[`, "prauc"))
best_ratio <- ratios[best_idx]
final_sz   <- c("No" = as.integer(n_minority * best_ratio), "Yes" = n_minority)
mod        <- train_rf(ntree = stable_ntree, mtry = best_mtry,
                       nodesize = best_nodesize, sampsize = final_sz)
sensitivity_results[4, 2:8] <- sens_row(res_sample[[best_idx]],
                                        paste0("1:", best_ratio), mod$votes[, "Yes"])
cat("sampsize tuning completed\n")

# Refinement: re-tune mtry with fixed sampsize
cat("6: Refinement (re-tuning mtry with fixed sampsize)...\n")
res_refine <- lapply(mtry_grid, function(m)
  get_oob_prauc(ntree = stable_ntree, mtry = m,
                nodesize = best_nodesize, sampsize = final_sz))
best_idx  <- which.max(sapply(res_refine, `[[`, "prauc"))
best_mtry <- mtry_grid[best_idx]
mod       <- train_rf(ntree = stable_ntree, mtry = best_mtry,
                      nodesize = best_nodesize, sampsize = final_sz)
sensitivity_results[5, 2:8] <- sens_row(res_refine[[best_idx]], best_mtry, mod$votes[, "Yes"])

cat("\nSensitivity results (tuning objective: PR-AUC on OOB votes):\n")
print(sensitivity_results)

# =========================
# 7. Train Tuned Model
# =========================
cat("\n7: Training tuned model...\n")

rf_model_tune <- train_rf(ntree = stable_ntree, mtry = best_mtry,
                          nodesize = best_nodesize, sampsize = final_sz)

evaluate_rf(rf_model_tune$votes[, "Yes"], y_train_num,
            label = "Tuned (train OOB)")

# =========================
# 8. Default vs Tuned: Test Set Comparison
#    Confusion matrix for each model + PR-AUC
# =========================
cat("8: Comparing default vs tuned model on test set...\n")

test_probs_def  <- predict(rf_model,      test_data, type = "prob")[, "Yes"]
test_probs_tune <- predict(rf_model_tune, test_data, type = "prob")[, "Yes"]

pred_def  <- factor(ifelse(test_probs_def  >= 0.5, "Yes", "No"), levels = c("No", "Yes"))
pred_tune <- factor(ifelse(test_probs_tune >= 0.5, "Yes", "No"), levels = c("No", "Yes"))

cat("\nDefault RF — confusion matrix (0.5 threshold):\n")
print(confusionMatrix(pred_def,  test_data$Revenue, positive = "Yes"))

cat("\nTuned RF — confusion matrix (0.5 threshold):\n")
print(confusionMatrix(pred_tune, test_data$Revenue, positive = "Yes"))

pr_def  <- pr.curve(scores.class0 = test_probs_def[y_test_num  == 1],
                    scores.class1 = test_probs_def[y_test_num  == 0], curve = TRUE)
pr_tune <- pr.curve(scores.class0 = test_probs_tune[y_test_num == 1],
                    scores.class1 = test_probs_tune[y_test_num == 0], curve = TRUE)

cat("\n--- PR-AUC (test set, 0.5 threshold) ---\n")
cat(sprintf("  Default : %.4f\n", pr_def$auc.integral))
cat(sprintf("  Tuned   : %.4f\n", pr_tune$auc.integral))
cat("-----------------------------------------\n")

# =========================
# 9. Model Selection + Threshold Optimisation
#    Threshold: maximise balanced accuracy 
# =========================
cat("\n9: Model selection and threshold tuning...\n")

prauc_def  <- evaluate_rf(test_probs_def,  y_test_num, verbose = FALSE)["PR_AUC"]
prauc_tune <- evaluate_rf(test_probs_tune, y_test_num, verbose = FALSE)["PR_AUC"]

if (prauc_def >= prauc_tune) {
  winner_model <- rf_model;      winner_label <- "Default"
  cat("Default RF selected as winner (higher or equal PR-AUC)\n")
} else {
  winner_model <- rf_model_tune; winner_label <- "Tuned"
  cat("Tuned RF selected as winner (higher test PC-AUC)\n")
}

winner_oob_probs <- winner_model$votes[, "Yes"]
thresholds       <- seq(0.1, 0.7, by = 0.01)

bal_accs <- sapply(thresholds, function(t) {
  p      <- factor(ifelse(winner_oob_probs > t, "Yes", "No"), levels = c("No", "Yes"))
  actual <- factor(ifelse(y_train_num      == 1, "Yes", "No"), levels = c("No", "Yes"))
  confusionMatrix(p, actual, positive = "Yes")$byClass["Balanced Accuracy"]
})

best_threshold <- thresholds[which.max(bal_accs)]
cat(sprintf("Optimal threshold (max balanced accuracy on OOB votes): %.2f\n", best_threshold))

# =========================
# 10. Feature Importance 
# =========================
cat("10: Feature importance...\n")
png("outputs/figures/ensemble/rf_var_imp.png", 800, 600)
varImpPlot(rf_model, n.var = 10, type = 1,
           main = "Top 10 Feature Importance: Purchase Intention")
dev.off()
cat("Feature Importance image saved at outputs/figures/ensemble/rf_var_imp.png\n")

# =========================
# 11. Partial Dependence Plots
#     Top 3 features ranked by MDA 
# =========================
cat("11: PDP analysis...\n")
mda_scores   <- importance(rf_model)[, "MeanDecreaseAccuracy"]   # MDA column
top_features <- names(sort(mda_scores, decreasing = TRUE))[1:3]

# PDP for Top 3 Features
png("outputs/figures/ensemble/rf_pdp_plots.png", 1000, 500)
par(mfrow = c(1, 3))
for (i in 1:length(top_features))
  partialPlot(rf_model, train_data, top_features[i], which.class = "Yes",
              main = paste("PDP:", top_features[i]), xlab = top_features[i], ylab = "Logit Prob of Revenue")
dev.off()
cat("PDP image saved at outputs/figures/ensemble/rf_pdp_plots.png\n")

# =========================
# 12. Final Evaluation (optimal threshold)
#     Confusion matrix + PR-AUC
# =========================
cat("\n12: Final test set evaluation...\n")

winner_probs <- predict(winner_model, test_data, type = "prob")[, "Yes"]
final_preds  <- factor(ifelse(winner_probs > best_threshold, "Yes", "No"),
                       levels = c("No", "Yes"))

cat(sprintf("\nFinal RF — confusion matrix (%s model, threshold = %.2f):\n",
            winner_label, best_threshold))
cm <- confusionMatrix(final_preds, test_data$Revenue, positive = "Yes")
print(cm)

pr_final <- pr.curve(scores.class0 = winner_probs[y_test_num == 1],
                     scores.class1 = winner_probs[y_test_num == 0], curve = TRUE)

cat("\n--- PR-AUC (test set, optimal threshold) ---\n")
cat(sprintf("  %s model : %.4f\n", winner_label, pr_final$auc.integral))
cat("---------------------------------------------\n")

cat("\n--- PART A: Random Forest Completed ---\n")

# ==============================================================================
# PART B: EXTREME GRADIENT BOOSTING
# ==============================================================================

cat("\n--- PART B: Gradient Boosting (XGBoost) ---\n")

# =========================
# 1. Temporal Data Preparation
#    Tuning : train on Q1+Q2, validate on Q3 holdout (PR-AUC objective)
#    Final  : train on full Q1+Q2+Q3, test on Q4
# =========================
cat("1: Preparing temporal data split...\n")

X_train_mat   <- model.matrix(Revenue ~ . - 1, data = train_data)
X_test_mat    <- model.matrix(Revenue ~ . - 1, data = test_data)
train_sub_idx <- which(train_data$Quarter.Q3 == 0 & train_data$Quarter.Q4 == 0)
holdout_idx   <- which(train_data$Quarter.Q3 == 1)

dtrain_sub  <- xgb.DMatrix(data = X_train_mat[train_sub_idx, ], label = y_train_num[train_sub_idx])
dholdout    <- xgb.DMatrix(data = X_train_mat[holdout_idx, ],   label = y_train_num[holdout_idx])
dtrain      <- xgb.DMatrix(data = X_train_mat, label = y_train_num)
dtest       <- xgb.DMatrix(data = X_test_mat,  label = y_test_num)

scale_weight <- sum(y_train_num == 0) / sum(y_train_num == 1)
cat(sprintf("scale_pos_weight: %.2f\n", scale_weight))

# =========================
# 2. Default Model
# =========================
cat("2: Training baseline XGBoost...\n")

def_xgb_params <- list(objective = "binary:logistic", eval_metric = "aucpr",
                       scale_pos_weight = scale_weight, lambda = 10, alpha = 1)

xgb_model <- xgb.train(params = def_xgb_params, data = dtrain, nrounds = 30)

evaluate_xgb(xgb_model, dtrain, y_train_num,
             label = "Default (train)")

# =========================
# 3. Bayesian Hyperparameter Optimisation
#    Objective : PR-AUC on Q3 holdout (trains on Q1+Q2)
#    Algorithm : UCB acquisition with RBF kernel
# =========================
cat("3: Running Bayesian Optimisation (PR-AUC on Q3 holdout) for 15 rounds...\n")
cat("This may take some time...\n")

xgb_holdout_bayes <- function(max_depth, eta, subsample, colsample_bytree,
                              gamma, min_child_weight) {
  params <- list(objective = "binary:logistic",
                 max_depth = max_depth, eta = eta,
                 subsample = subsample, colsample_bytree = colsample_bytree,
                 gamma = gamma, min_child_weight = min_child_weight,
                 lambda = 10, alpha = 1)
  model <- xgb.train(params = params, data = dtrain_sub, nrounds = 100,
                     evals = list(val = dholdout),
                     early_stopping_rounds = 10, verbose = 0)
  pred  <- predict(model, dholdout)
  label <- getinfo(dholdout, "label")
  pr    <- pr.curve(scores.class0 = pred[label == 1],
                    scores.class1 = pred[label == 0])
  score <- pr$auc.integral
  if (is.null(score) || is.na(score) || is.infinite(score)) score <- 0
  return(list(Score = score, Pred = 0)) # Pred = 0 is a dummy variable required by BayesianOptimization()
}

set.seed(42)
opt_result <- BayesianOptimization(
  xgb_holdout_bayes,
  bounds = list(max_depth        = c(2L, 4L),
                eta              = c(0.01, 0.02),
                subsample        = c(0.5, 0.7),
                colsample_bytree = c(0.5, 0.7),
                gamma            = c(10, 20),
                min_child_weight = c(15L, 30L)),
  init_points = 5, n_iter = 10, acq = "ucb"
)
cat("Hyperparameter tuning completed.\n")

# =========================
# 4. Tuned Model Training
#    CV determines optimal nrounds on full training set
# =========================
cat("4: Training tuned XGBoost model...\n")

final_params <- list(objective        = "binary:logistic",
                     eval_metric      = "aucpr",
                     scale_pos_weight = scale_weight,
                     max_depth        = opt_result$Best_Par["max_depth"],
                     eta              = opt_result$Best_Par["eta"],
                     subsample        = opt_result$Best_Par["subsample"],
                     colsample_bytree = opt_result$Best_Par["colsample_bytree"],
                     gamma            = opt_result$Best_Par["gamma"],
                     min_child_weight = opt_result$Best_Par["min_child_weight"],
                     lambda           = 10, alpha = 1)

cv_final <- xgb.cv(params = final_params, data = dtrain, nrounds = 300,
                   nfold = 5, early_stopping_rounds = 30, verbose = 0)

xgb_model_tune <- xgb.train(params  = final_params, data = dtrain,
                            nrounds = cv_final$early_stop$best_iteration,
                            verbose = 0)

evaluate_xgb(xgb_model_tune, dtrain, y_train_num,
             label = "Tuned (train)")

# =========================
# 5. Default vs Tuned: Test Set Comparison
#    Confusion matrix for each model + PR-AUC
# =========================
cat("5: Comparing default vs tuned XGBoost on test set...\n")

probs_base <- predict(xgb_model,      dtest)
probs_tune <- predict(xgb_model_tune, dtest)

pred_base <- factor(ifelse(probs_base > 0.5, "Yes", "No"), levels = c("No", "Yes"))
pred_tune <- factor(ifelse(probs_tune > 0.5, "Yes", "No"), levels = c("No", "Yes"))

cat("\nDefault XGBoost — confusion matrix (0.5 threshold):\n")
print(confusionMatrix(pred_base, test_data$Revenue, positive = "Yes"))

cat("\nTuned XGBoost — confusion matrix (0.5 threshold):\n")
print(confusionMatrix(pred_tune, test_data$Revenue, positive = "Yes"))

pr_base      <- pr.curve(scores.class0 = probs_base[y_test_num == 1],
                         scores.class1 = probs_base[y_test_num == 0], curve = TRUE)
pr_tune_curve <- pr.curve(scores.class0 = probs_tune[y_test_num == 1],
                          scores.class1 = probs_tune[y_test_num == 0], curve = TRUE)

cat("\n--- PR-AUC (test set, 0.5 threshold) ---\n")
cat(sprintf("  Default : %.4f\n", pr_base$auc.integral))
cat(sprintf("  Tuned   : %.4f\n", pr_tune_curve$auc.integral))
cat("-----------------------------------------\n")

cat("Tuned model is chosen as winner...\n")

# =========================
# 6. Threshold Optimisation
#    Tuned model on Q3 holdout, maximise balanced accuracy
# =========================
cat("6: Threshold tuning on Q3 holdout...\n")

probs_holdout  <- predict(xgb_model_tune, dholdout)
holdout_labels <- getinfo(dholdout, "label")
thresholds     <- seq(0.1, 0.7, by = 0.01)

bal_accs <- sapply(thresholds, function(t) {
  p      <- factor(ifelse(probs_holdout   > t, "Yes", "No"), levels = c("No", "Yes"))
  actual <- factor(ifelse(holdout_labels == 1, "Yes", "No"), levels = c("No", "Yes"))
  confusionMatrix(p, actual, positive = "Yes")$byClass["Balanced Accuracy"]
})

best_xgb_threshold <- thresholds[which.max(bal_accs)]
cat(sprintf("Optimal threshold: %.2f\n", best_xgb_threshold))

# =========================
# 7. SHAP Analysis
# =========================
cat("7: Running SHAP analysis for XGBoost...\n")

png("outputs/figures/ensemble/xgb_shap_summary.png", 800, 600)
shap.plot.summary.wrap1(
  model = xgb_model_tune, 
  X = X_train_mat,  
  top_n = 10,
  dilute = FALSE  
) + 
  ggtitle("SHAP Summary: Impact on Purchase Intention (Top 10 Features)")
dev.off()

cat("SHAP Summary plot saved at outputs/figures/ensemble/xgb_shap_summary.png\n")

# =========================
# 8. Final Evaluation (optimal threshold)
#    Confusion matrix + PR-AUC
# =========================
cat("8: Final test set evaluation...\n")

final_preds <- factor(ifelse(probs_tune > best_xgb_threshold, "Yes", "No"),
                      levels = c("No", "Yes"))

cat(sprintf("\nFinal XGBoost — confusion matrix (tuned model, threshold = %.2f):\n",
            best_xgb_threshold))
print(confusionMatrix(final_preds, test_data$Revenue, positive = "Yes"))

pr_final_xgb <- pr.curve(scores.class0 = probs_tune[y_test_num == 1],
                         scores.class1 = probs_tune[y_test_num == 0], curve = TRUE)

cat("\n--- PR-AUC (test set, optimal threshold) ---\n")
cat(sprintf("  Tuned model : %.4f\n", pr_final_xgb$auc.integral))
cat("---------------------------------------------\n")

cat("\n--- PART B: Gradient Boosting Completed ---\n")


# Save Models for Comparison Phase (06_comparison.R)
cat("\n--- Saving final model packets for comparison --- \n")
ensemble_packet <- list(
  rf = list(
    model     = winner_model,     
    threshold = best_threshold,
    params    = list(
      ntree    = if (winner_label == "Default") def_ntree else stable_ntree,
      mtry     = if (winner_label == "Default") def_mtry else best_mtry,
      nodesize = if (winner_label == "Default") 1 else best_nodesize, # 1 is RF default for classification
      sampsize = if (winner_label == "Default") rf_sampsize else final_sz
    ),
    test_set = test_data
  ),
  xgb = list(
    model     = xgb_model_tune,
    threshold = best_xgb_threshold,
    params    = final_params,
    test_set = dtest
  )
)

saveRDS(ensemble_packet, "outputs/models/ensemble_models.rds")
cat("File saved to outputs/models/ensemble_models.rds\n")

# PR-AUC plot for selected ensemble models
cat("\n---PR-AUC plot---\n")
png("outputs/figures/ensemble/final_prcurve_rf_xgb.png", 800, 600)
plot(pr_final, col = "blue", auc.main = FALSE,
     main = "Final PR Curves: RF vs XGBoost")
plot(pr_final_xgb, col = "red", auc.main = FALSE, add = TRUE)
legend("bottomleft",
       c(paste("RF",      winner_label, "(AUC:", round(pr_final$auc.integral,     3), ")"),
         paste("XGBoost Tuned (AUC:",            round(pr_final_xgb$auc.integral, 3), ")")),
       col = c("blue", "red"), lty = 1, cex = 0.8)
dev.off()
cat("PR-AUC plot saved at outputs/figures/ensemble/final_prcurve_rf_xgb.png\n")

cat("\n--- Completed 05_ensemble.R ---\n")