# ==============================================================================
# Script: 03_discriminative.R
# Author: Lau Jin Yi (2302313)
# Tasks: 
# 1. Regularized Logistic Regression (glmnet)
# 2. Classification Tree (rpart)
# 3. Class Imbalance Handling (Weights + Threshold Tuning)
# 4. Evaluation (Confusion Matrix, ROC/AUC)
# ==============================================================================

cat("\n--- Running 03_discriminative.R ---\n")

# 1. Load Required Libraries
library(glmnet)      # For regularized logistic regression
library(rpart)       # For CART
library(rpart.plot)  # For tree visualization
library(pROC)        # For ROC curves and threshold tuning
library(caret)       # For Confusion Matrix
library(PRROC)       # For Precision-Recall Curves

# Ensure output directories exist
if(!dir.exists("outputs/figures/discriminative")) dir.create("outputs/figures/discriminative", recursive = TRUE)
if(!dir.exists("outputs/models")) dir.create("outputs/models", recursive = TRUE)

# 2. Load the Pre-processed Data
train_data <- readRDS("outputs/data/train_data.rds")
test_data  <- readRDS("outputs/data/test_data.rds")

# Create numeric targets for PRROC evaluation
y_train_num <- ifelse(train_data$Revenue == "Yes", 1, 0)
y_test_num  <- ifelse(test_data$Revenue  == "Yes", 1, 0)

cat("\nHandling Multicollinearity for Non-Tree Models...\n")

# Drop BounceRates due to high correlated with ExitRates
cat("Dropping BounceRates due to high correlation with ExitRates...\n")
train_data$BounceRates <- NULL
test_data$BounceRates <- NULL

# Engineer composite features
cat("Engineering composite features...\n")
safe_div <- function(a, b) {
  ifelse(b==0, 0, a / b)
}

train_data$avg_time_per_page_Administrative <- safe_div(train_data$Administrative_Duration, train_data$Administrative)
train_data$avg_time_per_page_Informational  <- safe_div(train_data$Informational_Duration,  train_data$Informational)
train_data$avg_time_per_page_ProductRelated <- safe_div(train_data$ProductRelated_Duration, train_data$ProductRelated)

test_data$avg_time_per_page_Administrative <- safe_div(test_data$Administrative_Duration, test_data$Administrative)
test_data$avg_time_per_page_Informational  <- safe_div(test_data$Informational_Duration,  test_data$Informational)
test_data$avg_time_per_page_ProductRelated <- safe_div(test_data$ProductRelated_Duration, test_data$ProductRelated)

# Remove original features
cat("Removing original features...\n")
cols_to_remove <- c(
  "Administrative", "Administrative_Duration",
  "Informational",  "Informational_Duration",
  "ProductRelated", "ProductRelated_Duration"
)

train_data[cols_to_remove] <- list(NULL)
test_data[cols_to_remove] <- list(NULL)

cat("-> Successfully handled multicollinearity.\n")

# Separate features (X) and target (y)
# Note: glmnet requires matrix format for X
X_train <- model.matrix(Revenue ~ . - 1, data = train_data)
y_train <- train_data$Revenue

X_test <- model.matrix(Revenue ~ . - 1, data = test_data)
y_test <- test_data$Revenue

# 3. Handle Class Imbalance via Case Weights
# Calculate weights: inversely proportional to class frequencies
class_freq <- table(y_train)
weights <- ifelse(y_train == "Yes", 
                  (1 / class_freq["Yes"]) * (length(y_train) / 2), 
                  (1 / class_freq["No"])  * (length(y_train) / 2))

cat("\n--- PART A: Regularized Logistic Regression ---\n")
nzv_cols <- nearZeroVar(X_train)
if(length(nzv_cols) > 0) {
  X_train <- X_train[, -nzv_cols]
  X_test  <- X_test[, -nzv_cols]
  cat("Removed NZV features for glmnet.\n")
}

set.seed(123)

# Train Elastic Net using cv.glmnet. Include the 'weights' argument.
logreg_model <- cv.glmnet(X_train, y_train, family = "binomial", alpha = 0.5, weights = weights)

# Predict probabilities on the training set using the best lambda
logreg_train_probs <- predict(logreg_model, newx = X_train, type = "response", s = "lambda.min")

# Threshold Tuning using ROC curve
logreg_roc <- roc(y_train, as.numeric(logreg_train_probs))
optimal_logreg_thresh <- coords(logreg_roc, "best", ret = "threshold")$threshold
cat(sprintf("Optimal Logistic Threshold: %.4f\n", optimal_logreg_thresh))

# Predict on test set, apply optimal threshold, and generate caret::confusionMatrix
logreg_test_probs <- predict(logreg_model, newx = X_test, type = "response", s = "lambda.min")

logreg_pred <- ifelse(logreg_test_probs > optimal_logreg_thresh, "Yes", "No")
logreg_pred <- factor(logreg_pred, levels = c("No", "Yes"))

logreg_cm <- confusionMatrix(logreg_pred, y_test)
print(logreg_cm)

# --- Interpretability: Compute PR-AUC (Ensemble Style) ---
pr_logreg <- pr.curve(scores.class0 = logreg_test_probs[y_test_num == 1],
                      scores.class1 = logreg_test_probs[y_test_num == 0], curve = TRUE)

cat("\n--- PR-AUC (Logistic Regression Test Set) ---\n")
cat(sprintf("  PR-AUC : %.4f\n", pr_logreg$auc.integral))
cat("---------------------------------------------\n")

# --- Interpretability: Extract and Output Odds Ratios ---
cat("\nExtracting Odds Ratios to identify strong drivers of purchase intention...\n")

# Extract coefficients at the optimal lambda
logreg_coefs <- coef(logreg_model, s = "lambda.min")

# Convert to a standard matrix and remove the intercept
coef_matrix <- as.matrix(logreg_coefs)
coef_matrix <- coef_matrix[rownames(coef_matrix) != "(Intercept)", , drop=FALSE]

# Calculate Odds Ratios (exp(coefficient))
# Exclude variables whose coefficient was shrunk to exactly 0 by Elastic Net
active_coefs <- coef_matrix[coef_matrix[,1] != 0, , drop=FALSE]
odds_ratios <- exp(active_coefs)

# Combine into a data frame and sort to find the strongest drivers
or_df <- data.frame(Feature = rownames(odds_ratios), OddsRatio = odds_ratios[,1])
or_df <- or_df[order(-or_df$OddsRatio), ]

cat("\nTop Positive Drivers (Odds Ratio > 1 increases likelihood):\n")
print(head(or_df, 10))

cat("\nTop Negative Drivers (Odds Ratio < 1 decreases likelihood):\n")
print(tail(or_df, 5))

cat("\n--- PART B: Classification Tree (CART) ---\n")

set.seed(123)

# Build full tree using rpart(). Include the 'weights' argument.
cart_model <- rpart(Revenue ~ ., data = train_data, method = "class", weights = weights, control = rpart.control(cp = 0.001))

# Find optimal Complexity Parameter (CP) and prune the tree
opt_cp <- cart_model$cptable[which.min(cart_model$cptable[,"xerror"]),"CP"]
pruned_cart <- prune(cart_model, cp = opt_cp)
cat(sprintf("Optimal CP: %.4f\n", opt_cp))

# Plot pruned tree and save to outputs/figures/discriminative/
png("outputs/figures/discriminative/cart_tree.png")
rpart.plot(pruned_cart)
dev.off()

# Predict probabilities on train set, perform threshold tuning via ROC
cart_train_probs <- predict(pruned_cart, train_data, type = "prob")[,2]

cart_roc <- roc(y_train, cart_train_probs)
optimal_cart_thresh <- coords(cart_roc, "best", ret = "threshold")$threshold
cat(sprintf("Optimal CART Threshold: %.4f\n", optimal_cart_thresh))

# Predict on test set, apply optimal threshold, and generate caret::confusionMatrix
cart_test_probs <- predict(pruned_cart, test_data, type = "prob")[,2]

cart_pred <- ifelse(cart_test_probs > optimal_cart_thresh, "Yes", "No")
cart_pred <- factor(cart_pred, levels = c("No", "Yes"))

cart_cm <- confusionMatrix(cart_pred, y_test)
print(cart_cm)

# --- Interpretability: Compute PR-AUC ---
pr_cart <- pr.curve(scores.class0 = cart_test_probs[y_test_num == 1],
                    scores.class1 = cart_test_probs[y_test_num == 0], curve = TRUE)

cat("\n--- PR-AUC (CART Test Set) ---\n")
cat(sprintf("  PR-AUC : %.4f\n", pr_cart$auc.integral))
cat("---------------------------------------------\n")

# Plot ROC curves for both models  and save to outputs/figures/discriminative/
png("outputs/figures/discriminative/roc_comparison.png", width=1200, height=800)
plot(roc(y_test, as.numeric(logreg_test_probs)), col="blue", main="ROC Curve Comparison")
lines(roc(y_test, cart_test_probs), col="red")
legend("bottomright", legend=c("Logistic", "CART"), col=c("blue", "red"), lwd=2)
dev.off()

# Plot PR curves for both models and save
png("outputs/figures/discriminative/pr_comparison.png", width=1200, height=800)
plot(pr_logreg, color="blue", auc.main=FALSE, main="PR Curve Comparison: Logistic vs CART")
plot(pr_cart, color="red", auc.main=FALSE, add=TRUE)
legend("bottomleft",
       c(paste("Logistic (AUC:", round(pr_logreg$auc.integral, 3), ")"),
         paste("CART     (AUC:", round(pr_cart$auc.integral, 3), ")")),
       col=c("blue", "red"), lty=1, lwd=2)
dev.off()
cat("PR Curve comparison saved at outputs/figures/discriminative/pr_comparison.png\n")

# 4. Save Models for Comparison Phase
saveRDS(list(logreg = logreg_model, cart = pruned_cart), "outputs/models/discriminative_models.rds")

cat("\n--- Completed 03_discriminative.R ---\n")