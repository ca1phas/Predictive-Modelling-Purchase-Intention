# ==============================================================================
# Script: 06_comparison.R
# Author: Kok Shau Loon (2401978)
# Tasks: 
# 1. Load the preserved test_data.rds
# 2. Load all saved supervised models from outputs/models/
# 3. Generate predictions and extract metrics (Accuracy, Kappa, AUC, etc.)
# 4. Compile the final "Model Comparison" table
# ==============================================================================

cat("\n--- Running 06_comparison.R ---\n")

# Silent loading since there are a lot of packages
suppressPackageStartupMessages({
  library(caret)          # confusionMatrix
  library(pROC)           # roc, auc
  library(PRROC)          # pr.curve  (PR-AUC)
  library(xgboost)        # predict.xgb.Booster
  library(randomForest)   # predict.randomForest
  library(MASS)           # lda predict
  library(e1071)          # naiveBayes predict
  library(glmnet)         # predict.cv.glmnet
  library(ggplot2)        # plotting
  library(scales)         # percent_format
  library(knitr)          # kable – pretty console table
})

# Output directories
dir.create("outputs/tables",             recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/figures/comparison", recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------------------------
# 1. Load test data
# ------------------------------------------------------------------------------
cat("Loading test data...\n")
test_data   <- readRDS("outputs/data/test_data.rds")
y_test      <- test_data$Revenue              # factor "No"/"Yes"
y_test_num  <- ifelse(y_test == "Yes", 1L, 0L)# integer 0/1

# Standardize postive class = Yes and negative class = No
pos_class   <- "Yes"; neg_class   <- "No"

# ------------------------------------------------------------------------------
# 2. Load saved model packets
# ------------------------------------------------------------------------------
cat("Loading saved model packets...\n")

# Helper function to load models if the file exists
load_if_exists <- function(path) {
  if (file.exists(path)) {
    readRDS(path)
  } else {
    warning(sprintf("File not found: %s", path))
    NULL
  }
}

disc_model <- load_if_exists("outputs/models/discriminative_models.rds") # list: $logreg, $cart
gen_model  <- load_if_exists("outputs/models/generative_models.rds")     # list: $nb, $lda
ens_model  <- load_if_exists("outputs/models/ensemble_models.rds")       # list: $rf, $xgb

# Guard: stop if nothing loaded
if (all(sapply(list(gen_model, ens_model, disc_model), is.null))) {
  stop("No model files found. Please run scripts 03–05 first.")
}

# ------------------------------------------------------------------------------
# 3. Shared helper: extract all metrics from prob scores + actuals
# ------------------------------------------------------------------------------
# Returns a named numeric vector:
#   Accuracy, Kappa, Sensitivity, Specificity, Precision, F1,
#   Balanced_Accuracy, PR_AUC, ROC_AUC
compute_metrics <- function(probs, actuals_num, threshold, model_name) {
  pred_labels  <- factor(ifelse(probs >= threshold, pos_class, neg_class),
                         levels = c(neg_class, pos_class))
  true_labels  <- factor(ifelse(actuals_num == 1, pos_class, neg_class),
                         levels = c(neg_class, pos_class))
  
  cm  <- confusionMatrix(pred_labels, true_labels, positive = pos_class)
  
  # PR-AUC via PRROC (scores.class0 = positives, scores.class1 = negatives)
  pr_obj <- pr.curve(scores.class0 = probs[actuals_num == 1],
                     scores.class1 = probs[actuals_num == 0],
                     curve = TRUE)
  
  # ROC-AUC via pROC
  roc_obj <- roc(response  = actuals_num,
                 predictor = probs,
                 levels    = c(0, 1),
                 direction = "<",
                 quiet     = TRUE)
  
  metrics <- c(
    Accuracy          = round(cm$overall["Accuracy"],              4),
    Kappa             = round(cm$overall["Kappa"],                 4),
    Sensitivity       = round(cm$byClass["Sensitivity"],           4),
    Specificity       = round(cm$byClass["Specificity"],           4),
    Precision         = round(cm$byClass["Pos Pred Value"],        4),
    F1                = round(cm$byClass["F1"],                    4),
    Balanced_Accuracy = round(cm$byClass["Balanced Accuracy"],     4),
    PR_AUC            = round(pr_obj$auc.integral,                 4),
    ROC_AUC           = round(as.numeric(auc(roc_obj)),            4)
  )
  
  # Also return curve objects for overlay plots (attach as attributes)
  attr(metrics, "pr_curve")  <- pr_obj
  attr(metrics, "roc_curve") <- roc_obj
  attr(metrics, "cm")        <- cm
  attr(metrics, "threshold") <- threshold
  attr(metrics, "name")      <- model_name
  
  return(metrics)
}

# ------------------------------------------------------------------------------
# 4. Generate predictions & metrics for every model
# ------------------------------------------------------------------------------
cat("Generating predictions and computing metrics...\n\n")

results_list   <- list()   # metrics vectors
pr_curves_list <- list()   # PRROC objects (for overlay plot)
roc_curves_list<- list()   # pROC objects  (for overlay plot)
cm_list        <- list()   # confusionMatrix objects

# ---- 4a. Elastic Net Logistic Regression (Model 1) --------------------------
if (!is.null(disc_model$logreg)) {
  cat("  [1/6] Elastic Net Logistic Regression...\n")
  
  lr_probs_raw <- predict(disc_model$logreg$model, newx = disc_model$logreg$test_set,
                          type = "response", s = "lambda.min")
  lr_probs     <- as.numeric(lr_probs_raw)
  
  lr_thresh    <- disc_model$logreg$threshold
  
  m <- compute_metrics(lr_probs, y_test_num, lr_thresh, "LogReg")
  results_list[["LogReg"]]    <- m
  pr_curves_list[["LogReg"]]  <- attr(m, "pr_curve")
  roc_curves_list[["LogReg"]] <- attr(m, "roc_curve")
  cm_list[["LogReg"]]         <- attr(m, "cm")
}

# ---- 4b. CART (Model 2) -----------------------------------------------------
if (!is.null(disc_model$cart)) {
  cat("  [2/6] CART...\n")
  cart_probs <- predict(disc_model$cart$model, newdata = disc_model$cart$test_set, type = "prob")[, pos_class]
  
  cart_thresh  <- disc_model$cart$threshold
  
  m <- compute_metrics(cart_probs, y_test_num, cart_thresh, "CART")
  results_list[["CART"]]    <- m
  pr_curves_list[["CART"]]  <- attr(m, "pr_curve")
  roc_curves_list[["CART"]] <- attr(m, "roc_curve")
  cm_list[["CART"]]         <- attr(m, "cm")
}

# ---- 4c. Naive Bayes (Model 3) -----------------------------------------------
if (!is.null(gen_model$nb)) {
  cat("  [3/6] Naive Bayes...\n")
  nb_probs <- predict(gen_model$nb$model, newdata = gen_model$nb$test_set, type = "raw")[, pos_class]
  
  nb_thresh    <- gen_model$nb$threshold
  
  m <- compute_metrics(nb_probs, y_test_num, nb_thresh, "Naive Bayes")
  results_list[["Naive Bayes"]]    <- m
  pr_curves_list[["Naive Bayes"]]  <- attr(m, "pr_curve")
  roc_curves_list[["Naive Bayes"]] <- attr(m, "roc_curve")
  cm_list[["Naive Bayes"]]         <- attr(m, "cm")
}

# ---- 4d. LDA (Model 4) -------------------------------------------------------
if (!is.null(gen_model$lda)) {
  cat("  [4/6] LDA...\n")
  lda_probs <- predict(gen_model$lda$model, newdata = gen_model$lda$test_set)$posterior[, pos_class]
  lda_thresh    <- gen_model$lda$threshold
  
  m <- compute_metrics(lda_probs, y_test_num, lda_thresh, "LDA")
  results_list[["LDA"]]    <- m
  pr_curves_list[["LDA"]]  <- attr(m, "pr_curve")
  roc_curves_list[["LDA"]] <- attr(m, "roc_curve")
  cm_list[["LDA"]]         <- attr(m, "cm")
}

# ---- 4e. Random Forest (Model 5) --------------------------------------------
if (!is.null(ens_model$rf)) {
  cat("  [5/6] Random Forest...\n")
  rf_probs  <- predict(ens_model$rf$model, newdata = ens_model$rf$test_set, type = "prob")[, pos_class]
  rf_thresh <- ens_model$rf$threshold  
  
  m <- compute_metrics(rf_probs, y_test_num, rf_thresh, "Random Forest")
  results_list[["Random Forest"]]    <- m
  pr_curves_list[["Random Forest"]]  <- attr(m, "pr_curve")
  roc_curves_list[["Random Forest"]] <- attr(m, "roc_curve")
  cm_list[["Random Forest"]]         <- attr(m, "cm")
}

# ---- 4f. XGBoost (Model 6) --------------------------------------------------
if (!is.null(ens_model$xgb)) {
  cat("  [6/6] XGBoost...\n")
  X_test_mat    <- model.matrix(Revenue ~ . - 1, data = test_data)
  dtest       <- xgb.DMatrix(data = X_test_mat,  label = y_test_num)
  
  xgb_probs  <- predict(ens_model$xgb$model, newdata = dtest)
  xgb_thresh <- ens_model$xgb$threshold
  
  m <- compute_metrics(xgb_probs, y_test_num, xgb_thresh, "XGBoost")
  results_list[["XGBoost"]]    <- m
  pr_curves_list[["XGBoost"]]  <- attr(m, "pr_curve")
  roc_curves_list[["XGBoost"]] <- attr(m, "roc_curve")
  cm_list[["XGBoost"]]         <- attr(m, "cm")
}

# ------------------------------------------------------------------------------
# 5. Build & print comparison table
# ------------------------------------------------------------------------------
cat("\n")

# Strip curve attributes before converting to data.frame
strip_attrs <- function(x) {
  attrs_to_keep <- names(x)
  # Remove everything after the first dot (including the dot)
  clean_names <- gsub("\\..*$", "", attrs_to_keep)
  out <- setNames(as.numeric(x), clean_names)
  out
}

comparison_df <- do.call(rbind, lapply(names(results_list), function(nm) {
  v <- strip_attrs(results_list[[nm]])
  thresh <- attr(results_list[[nm]], "threshold")
  data.frame(
    Model             = nm,
    Threshold         = round(thresh, 4),
    Accuracy          = v["Accuracy"],
    Kappa             = v["Kappa"],
    Sensitivity       = v["Sensitivity"],
    Specificity       = v["Specificity"],
    Precision         = v["Precision"],
    F1                = v["F1"],
    Balanced_Accuracy = v["Balanced_Accuracy"],
    PR_AUC            = v["PR_AUC"],
    ROC_AUC           = v["ROC_AUC"],
    stringsAsFactors  = FALSE,
    row.names         = NULL
  )
}))

# Sort by PR-AUC (primary) then Balanced Accuracy (tie-break)
comparison_df <- comparison_df[order(-comparison_df$PR_AUC,
                                     -comparison_df$Balanced_Accuracy), ]

cat("=======================================================================\n")
cat("                   FINAL MODEL COMPARISON TABLE                        \n")
cat("  (sorted by PR-AUC descending — primary metric for imbalanced data)   \n")
cat("=======================================================================\n\n")
print(kable(comparison_df, format = "simple", digits = 4, row.names = FALSE))

# Save CSV
write.csv(comparison_df, "outputs/tables/final_model_comparison.csv",
          row.names = FALSE)
cat("\n-> Saved: outputs/tables/final_model_comparison.csv\n")

# ------------------------------------------------------------------------------
# 6. Identify best model
# ------------------------------------------------------------------------------
best_model_name <- comparison_df$Model[1]
best_row        <- comparison_df[1, ]

cat("\n=======================================================================\n")
cat(sprintf("  BEST MODEL: %s\n", best_model_name))
cat(sprintf("  PR-AUC            : %.4f\n", best_row$PR_AUC))
cat(sprintf("  ROC-AUC           : %.4f\n", best_row$ROC_AUC))
cat(sprintf("  Balanced Accuracy : %.4f\n", best_row$Balanced_Accuracy))
cat(sprintf("  Sensitivity       : %.4f\n", best_row$Sensitivity))
cat(sprintf("  Specificity       : %.4f\n", best_row$Specificity))
cat(sprintf("  Precision         : %.4f\n", best_row$Precision))
cat(sprintf("  F1                : %.4f\n", best_row$F1))
cat(sprintf("  Kappa             : %.4f\n", best_row$Kappa))
cat("=======================================================================\n\n")

# ------------------------------------------------------------------------------
# 7. Visualisations
# ------------------------------------------------------------------------------
cat("Generating comparison plots...\n")

model_colors <- c(
  "LogReg"         = "#E41A1C",
  "Naive Bayes"    = "#FF7F00",
  "CART"           = "#4DAF4A",
  "LDA"            = "#377EB8",
  "Random Forest"  = "#984EA3",
  "XGBoost"        = "#A65628"
)

# ---- 7a. PR Curve comparison ---------------------------------------------------
png("outputs/figures/comparison/pr_curve_comparison.png", width = 800, height = 600)
par(mar = c(5, 5, 4, 10), xpd = FALSE)
plot(NULL, xlim = c(0, 1), ylim = c(0, 1),
     xlab = "Recall", ylab = "Precision",
     main = "Precision-Recall Curve Comparison (All Models)")
abline(h = mean(y_test_num), lty = 2, col = "grey60")  # baseline

for (nm in names(pr_curves_list)) {
  pr <- pr_curves_list[[nm]]
  if (!is.null(pr$curve)) {
    lines(pr$curve[, 1], pr$curve[, 2],
          col = model_colors[nm], lwd = 2)
  }
}
legend(x = "bottomleft", y = 1,
       legend = sapply(names(pr_curves_list), function(nm)
         sprintf("%s (%.3f)", nm, pr_curves_list[[nm]]$auc.integral)),
       col    = model_colors[names(pr_curves_list)],
       lwd = 2, bty = "n", cex = 1.75)
dev.off()
cat("  -> outputs/figures/comparison/pr_curve_comparison.png\n")

# ---- 7b. ROC Curve comparison --------------------------------------------------
png("outputs/figures/comparison/roc_curve_comparison.png", width = 800, height = 600)
par(mar = c(5, 5, 4, 10), xpd = FALSE)
plot(roc_curves_list[[1]], col = model_colors[names(roc_curves_list)[1]],
     lwd = 2, main = "ROC Curve Comparison (All Models)")
for (i in seq_along(roc_curves_list)[-1]) {
  nm <- names(roc_curves_list)[i]
  plot(roc_curves_list[[nm]], add = TRUE, col = model_colors[nm], lwd = 2)
}
abline(a = 0, b = 1, lty = 2, col = "grey60")
legend(x = "bottomright", y = 1,
       legend = sapply(names(roc_curves_list), function(nm)
         sprintf("%s (%.3f)", nm, as.numeric(auc(roc_curves_list[[nm]])))),
       col    = model_colors[names(roc_curves_list)],
       lwd = 2, bty = "n", cex = 1.75)
dev.off()
cat("  -> outputs/figures/comparison/roc_curve_comparison.png\n")

# ---- 7c. Radar / spider chart (base-R version) ------------------------------
# Metrics normalised to [0,1] for radar
radar_metrics <- c("PR_AUC", "ROC_AUC", "Balanced_Accuracy",
                   "F1", "Sensitivity", "Specificity", "Precision")
radar_df <- comparison_df[, c("Model", radar_metrics)]
rownames(radar_df) <- radar_df$Model

n_axis   <- length(radar_metrics)
angles   <- seq(0, 2 * pi, length.out = n_axis + 1)[-(n_axis + 1)]

png("outputs/figures/comparison/radar_chart.png", width = 800, height = 800)
par(mar = c(2, 2, 4, 2))
plot(NULL, xlim = c(-1.4, 1.4), ylim = c(-1.4, 1.4),
     asp = 1, axes = FALSE, xlab = "", ylab = "",
     main = "Radar Chart: Model Performance Profile")

# Draw grid rings at 0.25, 0.5, 0.75, 1.0
for (r in c(0.25, 0.5, 0.75, 1.0)) {
  polygon(r * cos(angles), r * sin(angles), border = "grey80", lty = 2)
  text(0, r + 0.03, sprintf("%.2f", r), cex = 1.05, col = "grey50")
}
# Axis spokes & labels
for (i in seq_len(n_axis)) {
  lines(c(0, cos(angles[i])), c(0, sin(angles[i])), col = "grey70")
  text(1.22 * cos(angles[i]), 1.22 * sin(angles[i]),
       gsub("_", "\n", radar_metrics[i]), cex = 1.1, font = 2)
}
# Draw each model polygon
model_names_radar <- radar_df$Model
for (nm in model_names_radar) {
  vals <- as.numeric(radar_df[nm, radar_metrics])
  x    <- c(vals * cos(angles), vals[1] * cos(angles[1]))
  y    <- c(vals * sin(angles), vals[1] * sin(angles[1]))
  col  <- model_colors[nm]
  lines(x, y, col = col, lwd = 2)
  points(vals * cos(angles), vals * sin(angles),
         col = col, pch = 19, cex = 1.3)
}
legend("bottomleft", legend = model_names_radar,
       col = model_colors[model_names_radar],
       lwd = 2, pch = 19, bty = "n", cex = 1.49)
dev.off()
cat("  -> outputs/figures/comparison/radar_chart.png\n")


# ------------------------------------------------------------------------------
# 8. Summary export
# ------------------------------------------------------------------------------
# Add rank column for report
comparison_df$Rank <- seq_len(nrow(comparison_df))
comparison_df      <- comparison_df[, c("Rank", setdiff(names(comparison_df), "Rank"))]

write.csv(comparison_df, "outputs/tables/final_model_comparison.csv", row.names = FALSE)
cat("\n-> Saved comparison table to outputs/tables/final_model_comparison.csv\n")

cat("\n=======================================================================\n")
cat("\n--- Completed 06_comparison.R ---\n")
cat(sprintf("  Best model : %s  (PR-AUC = %.4f)\n",
            best_model_name, best_row$PR_AUC))
