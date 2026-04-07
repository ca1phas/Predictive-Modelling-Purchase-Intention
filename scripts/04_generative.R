# ==============================================================================
# Script: 04_generative.R
# Author: Chia Inn Xin (2202927)
# Tasks:
# 1. Base R Data Processing (Handling Multicollinearity & Near-Zero Variance)
# 2. Naive Bayes Classifier with Laplace Smoothing (e1071)
# 3. Linear Discriminant Analysis with Balanced Priors (MASS)
# 4. Class Imbalance Handling (Optimal Threshold Tuning via Youden's Index)
# 5. Advanced Evaluation & Visualizations (CM, ROC/AUC, PR Curve, Plots)
# ==============================================================================

cat("\n--- Running 04_generative.R ---\n")

# 1. Load Required Libraries
library(e1071) # For Naive Bayes
library(MASS) # For LDA
library(pROC) # For ROC curves and threshold tuning
library(caret) # For Confusion Matrix metrics
library(PRROC) # For Precision-Recall (PR) Curves

# Ensure output directories exist for saving models and figures
if (!dir.exists("outputs/figures/generative")) {
  dir.create("outputs/figures/generative", recursive = TRUE)
}
if (!dir.exists("outputs/models")) {
  dir.create("outputs/models", recursive = TRUE)
}

# 2. Load the Pre-processed Data
train_data <- readRDS("outputs/data/train_data.rds")
test_data <- readRDS("outputs/data/test_data.rds")

# 3. Feature Engineering & Multicollinearity Handling
cat("Handling multicollinearity and engineering composite features...\n")

# Create composite feature: average time spent per Administrative page
# Using ifelse to prevent division by zero if Administrative pages = 0
train_data$avg_time_Admin <- ifelse(
  train_data$Administrative == 0,
  0,
  train_data$Administrative_Duration / train_data$Administrative
)

test_data$avg_time_Admin <- ifelse(
  test_data$Administrative == 0,
  0,
  test_data$Administrative_Duration / test_data$Administrative
)

# Drop highly correlated features to prevent multicollinearity in LDA/NB
# Note: BounceRates is a subset of ExitRates, so we drop BounceRates
cols_to_drop <- c("Administrative", "Administrative_Duration", "BounceRates")

train_data <- train_data[, !(names(train_data) %in% cols_to_drop)]
test_data <- test_data[, !(names(test_data) %in% cols_to_drop)]

# Check and remove near-zero variance (NZV) predictors
# This prevents singular matrix / perfect separation errors during modeling
nzv_cols <- nearZeroVar(train_data, saveMetrics = TRUE)
nzv_names <- rownames(nzv_cols[nzv_cols$nzv == TRUE, ])

nzv_cols_to_drop <- c(nzv_names)
if (length(nzv_cols_to_drop) > 0) {
  cat("Dropping near-zero variance predictors to prevent errors...\n")
  train_data <- train_data[, !(names(train_data) %in% nzv_cols_to_drop)]
  test_data <- test_data[, !(names(test_data) %in% nzv_cols_to_drop)]
}

# PART A: Naive Bayes Classifier
cat("\n--- PART A: Naive Bayes ---\n")

# Identify the positive and negative classes dynamically
pos_class <- levels(train_data$Revenue)[2]
neg_class <- levels(train_data$Revenue)[1]

# Train Naive Bayes model
# Applied Laplace smoothing (laplace = 1) to handle zero-probability issues
cat("Training Naive Bayes model with Laplace smoothing...\n")
nb_model <- naiveBayes(Revenue ~ ., data = train_data, laplace = 1)

# Extract raw probabilities for the training set
nb_train_probs <- predict(nb_model,
  newdata = train_data,
  type = "raw"
)[, pos_class]

# Threshold Tuning: Use ROC curve to handle the severe class imbalance
cat("Tuning threshold using ROC curve to handle class imbalance...\n")
nb_roc <- roc(train_data$Revenue,
  nb_train_probs,
  levels = c(neg_class, pos_class),
  direction = "<",
  quiet = TRUE
)

# Find the optimal cut-off point using Youden's Index
optimal_nb_thresh <- coords(nb_roc,
  "best",
  ret = "threshold",
  best.method = "youden"
)$threshold

cat(sprintf(
  "Optimal Probability Threshold for Naive Bayes: %.4f\n",
  optimal_nb_thresh
))

# Predict on test set and apply the optimized threshold
nb_test_probs <- predict(nb_model,
  newdata = test_data,
  type = "raw"
)[, pos_class]

nb_pred_class <- ifelse(nb_test_probs >= optimal_nb_thresh,
  pos_class,
  neg_class
)
nb_pred_class <- factor(nb_pred_class, levels = levels(test_data$Revenue))

# Generate Confusion Matrix to evaluate model performance on test set
cat("\nNaive Bayes Evaluation on Test Set:\n")
nb_cm <- confusionMatrix(nb_pred_class,
  test_data$Revenue,
  positive = pos_class
)
print(nb_cm)

# --- Interpretability: Extract Naive Bayes Conditional Probabilities ---
cat("\nExtracting Naive Bayes Conditional Means to identify purchase intent drivers...\n")

# Initialize an empty data frame to store the results
nb_insights <- data.frame(Feature = character(), 
                          Mean_No = numeric(), 
                          Mean_Yes = numeric(), 
                          Ratio_Yes_vs_No = numeric(), 
                          stringsAsFactors = FALSE)

# Loop through each feature in the Naive Bayes tables
for (feat_name in names(nb_model$tables)) {
  feat_stats <- nb_model$tables[[feat_name]]
  
  # e1071 stores Mean in [,1] and SD in [,2] for continuous/scaled variables
  if(ncol(feat_stats) >= 1) {
    mean_no <- feat_stats["No", 1]
    mean_yes <- feat_stats["Yes", 1]
    
    # Calculate ratio (adding a small epsilon to prevent division by zero)
    ratio <- mean_yes / (mean_no + 1e-9)
    
    nb_insights <- rbind(nb_insights, data.frame(
      Feature = feat_name,
      Mean_No = round(mean_no, 4),
      Mean_Yes = round(mean_yes, 4),
      Ratio_Yes_vs_No = round(ratio, 4)
    ))
  }
}

# Sort by the highest ratio (strongest indicators for 'Yes')
nb_insights <- nb_insights[order(-nb_insights$Ratio_Yes_vs_No), ]

cat("\nTop 10 Strongest Indicators for Purchase Intent (Revenue = Yes):\n")
print(head(nb_insights, 10))
cat("--------------------------------------------------------------\n")

# PART B: Linear Discriminant Analysis (LDA)
cat("\n--- PART B: Linear Discriminant Analysis (LDA) ---\n")

# Define balanced priors (50/50)
balanced_priors <- c(0.5, 0.5)

# Train LDA model using balanced priors
cat("Training LDA model with balanced priors...\n")
lda_model <- lda(Revenue ~ ., data = train_data, prior = balanced_priors)

# Extract posterior probabilities for the training set
lda_train_probs <- predict(lda_model,
  newdata = train_data
)$posterior[, pos_class]

# Threshold Tuning: Find optimal cut-off for LDA
cat("Tuning threshold for LDA using ROC curve...\n")
lda_roc <- roc(train_data$Revenue,
  lda_train_probs,
  levels = c(neg_class, pos_class),
  direction = "<",
  quiet = TRUE
)

optimal_lda_thresh <- coords(lda_roc,
  "best",
  ret = "threshold",
  best.method = "youden"
)$threshold

cat(sprintf(
  "Optimal Probability Threshold for LDA: %.4f\n",
  optimal_lda_thresh
))

# Predict on test set and apply optimized threshold
lda_test_probs <- predict(lda_model,
  newdata = test_data
)$posterior[, pos_class]

lda_pred_class <- ifelse(lda_test_probs >= optimal_lda_thresh,
  pos_class,
  neg_class
)
lda_pred_class <- factor(lda_pred_class, levels = levels(test_data$Revenue))

# Generate Confusion Matrix to evaluate model performance
cat("\nLDA Evaluation on Test Set:\n")
lda_cm <- confusionMatrix(lda_pred_class,
  test_data$Revenue,
  positive = pos_class
)
print(lda_cm)

# PART C: Generating Visualizations
cat("\n--- PART C: Generating Visualizations ---\n")

# 1. Plot ROC comparison (Evaluates Trade-off between Sensitivity & Specificity)
cat("Generating ROC comparison plot...\n")
png("outputs/figures/generative/roc_comparison.png",
  width = 800, height = 600, res = 120
)
plot(nb_roc,
  col = "blue",
  main = "ROC Curve Comparison: Naive Bayes vs LDA", lwd = 2
)
plot(lda_roc, col = "red", add = TRUE, lwd = 2)
legend("bottomright",
  legend = c(
    sprintf("Naive Bayes (AUC = %.3f)", auc(nb_roc)),
    sprintf("LDA (AUC = %.3f)", auc(lda_roc))
  ),
  col = c("blue", "red"), lwd = 2, bty = "n"
)
dev.off()

# 2. Plot PR Curve comparison (Better metric for highly imbalanced datasets)
cat("Generating PR Curve comparison plot...\n")
actual_labels <- ifelse(test_data$Revenue == pos_class, 1, 0)

nb_pr <- pr.curve(
  scores.class0 = nb_test_probs,
  weights.class0 = actual_labels,
  curve = TRUE
)

lda_pr <- pr.curve(
  scores.class0 = lda_test_probs,
  weights.class0 = actual_labels,
  curve = TRUE
)

png("outputs/figures/generative/pr_comparison.png",
  width = 800, height = 600, res = 120
)
plot(lda_pr,
  col = "red",
  main = "PR Curve Comparison: Naive Bayes vs LDA", auc.main = FALSE
)
plot(nb_pr, col = "blue", add = TRUE)
legend("bottomleft",
  legend = c(
    sprintf("LDA (AUC-PR = %.3f)", lda_pr$auc.integral),
    sprintf("Naive Bayes (AUC-PR = %.3f)", nb_pr$auc.integral)
  ),
  col = c("red", "blue"), lwd = 2, bty = "n"
)
dev.off()

# 3. Plot LDA Projection (Visualizes linear separation between classes)
cat("Generating LDA Projection plot...\n")
png("outputs/figures/generative/lda_projection.png",
  width = 800, height = 600, res = 120
)
plot(lda_model,
  col = c("pink", "lightblue"),
  main = "LDA Projection: Buyer vs Non-Buyer"
)
dev.off()

# 4. Plot Fourfold Confusion Matrices (Visual representation of Accuracy)
cat("Generating Fourfold plots for Confusion Matrices...\n")
png("outputs/figures/generative/cm_fourfold_comparison.png",
  width = 800, height = 400, res = 120
)
par(mfrow = c(1, 2))
fourfoldplot(nb_cm$table,
  color = c("red", "green"),
  conf.level = 0, margin = 1, main = "Naive Bayes CM"
)
fourfoldplot(lda_cm$table,
  color = c("red", "green"),
  conf.level = 0, margin = 1, main = "LDA CM"
)
dev.off()

cat("All plots saved successfully to outputs/figures/generative/\n")

# --- Explicit Output for Model Comparison Table ---
cat("\n--- Final Generative Metrics for Comparison Table ---\n")
# Note: Ensure these variable names match what was defined in your ROC/PR calculations
cat(sprintf("Naive Bayes - AUC: %.4f | PR-AUC: %.4f\n", as.numeric(auc(nb_roc)), nb_pr$auc.integral))
cat(sprintf("LDA         - AUC: %.4f | PR-AUC: %.4f\n", as.numeric(auc(lda_roc)), lda_pr$auc.integral))
cat("-----------------------------------------------------\n")

# Save Models for Comparison Phase
cat("\nSaving trained Generative models to outputs/models/...\n")
gen_packet <- list(
  nb = list(
    model = nb_model,
    threshold = optimal_nb_thresh,
    test_set = test_data
  ),
  lda = list(
    model = lda_model,
    threshold = optimal_lda_thresh,
    test_set = test_data
  )
)

saveRDS(gen_packet,"outputs/models/generative_models.rds")

cat("\n--- Completed 04_generative.R ---\n")
