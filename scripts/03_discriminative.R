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

# Ensure output directories exist
if(!dir.exists("outputs/figures/discriminative")) dir.create("outputs/figures/discriminative", recursive = TRUE)
if(!dir.exists("outputs/models")) dir.create("outputs/models", recursive = TRUE)

# 2. Load the Pre-processed Data
train_data <- readRDS("outputs/data/train_data.rds")
test_data  <- readRDS("outputs/data/test_data.rds")

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
# Note: Check for perfect separation. If glmnet throws errors, manually drop NZV columns from X_train/X_test here.

# [TODO]: Train Ridge/Lasso/Elastic Net using cv.glmnet. Include the 'weights' argument.
# logreg_model <- cv.glmnet(X_train, y_train, family = "binomial", alpha = 1, weights = weights)

# [TODO]: Predict probabilities on the training set using the best lambda
# logreg_train_probs <- predict(logreg_model, newx = X_train, type = "response", s = "lambda.min")

# [TODO]: Threshold Tuning using ROC curve
# logreg_roc <- roc(y_train, as.numeric(logreg_train_probs))
# optimal_logreg_thresh <- coords(logreg_roc, "best", ret = "threshold")$threshold

# [TODO]: Predict on test set, apply optimal threshold, and generate caret::confusionMatrix

cat("\n--- PART B: Classification Tree (CART) ---\n")

# [TODO]: Build full tree using rpart(). Include the 'weights' argument.
# cart_model <- rpart(Revenue ~ ., data = train_data, method = "class", weights = weights, control = rpart.control(cp = 0.001))

# [TODO]: Find optimal Complexity Parameter (CP) and prune the tree
# opt_cp <- cart_model$cptable[which.min(cart_model$cptable[,"xerror"]),"CP"]
# pruned_cart <- prune(cart_model, cp = opt_cp)

# [TODO]: Plot pruned tree and save to outputs/figures/discriminative/
# png("outputs/figures/discriminative/cart_tree.png")
# rpart.plot(pruned_cart)
# dev.off()

# [TODO]: Predict probabilities on train set, perform threshold tuning via ROC
# [TODO]: Predict on test set, apply optimal threshold, and generate caret::confusionMatrix

# [TODO]: Plot ROC curves for both models  and save to outputs/figures/discriminative/

# 4. Save Models for Comparison Phase
# saveRDS(list(logreg = logreg_model, cart = pruned_cart), "outputs/models/discriminative_models.rds")

cat("\n--- Completed 03_discriminative.R ---\n")