# ==============================================================================
# Script: 04_generative.R
# Author: Chia Inn Xin (2202927)
# Tasks: 
# 1. Naive Bayes Classifier (e1071)
# 2. Linear Discriminant Analysis (MASS)
# 3. Class Imbalance Handling (Balanced Priors c(0.5, 0.5) + Threshold Tuning)
# 4. Evaluation (Confusion Matrix, ROC/AUC)
# ==============================================================================

cat("\n--- Running 04_generative.R ---\n")

# 1. Load Required Libraries
library(e1071)       # For Naive Bayes
library(MASS)        # For LDA
library(pROC)        # For ROC curves and threshold tuning
library(caret)       # For Confusion Matrix metrics

# Ensure output directories exist
if(!dir.exists("outputs/figures/generative")) dir.create("outputs/figures/generative", recursive = TRUE)
if(!dir.exists("outputs/models")) dir.create("outputs/models", recursive = TRUE)

# 2. Load the Pre-processed Data
train_data <- readRDS("outputs/data/train_data.rds")
test_data  <- readRDS("outputs/data/test_data.rds")

# 3. Data Handling: Check and remove near-zero variance predictors for Generative Models
# (Specifically handling singular matrix / perfect separation errors)
nzv_cols <- nearZeroVar(train_data, saveMetrics = TRUE)
nzv_names <- rownames(nzv_cols[nzv_cols$nzv == TRUE, ])

# dropping OperatingSystems.Other if it exists and causes singular matrix error
# cols_to_drop <- c(nzv_names) 
# if(length(cols_to_drop) > 0) {
#   cat("Dropping near-zero variance predictors to prevent singular matrix errors...\n")
#   train_data <- train_data[, !(names(train_data) %in% cols_to_drop)]
#   test_data <- test_data[, !(names(test_data) %in% cols_to_drop)]
# }

# Define the balanced priors
balanced_priors <- c(0.5, 0.5)

cat("\n--- PART A: Naive Bayes ---\n")

# [TODO]: Train Naive Bayes model using naiveBayes(). Include 'prior = balanced_priors'.
# nb_model <- naiveBayes(Revenue ~ ., data = train_data) # Note: e1071 handles priors via class frequencies natively or laplace smoothing, so you may need to adjust prediction logic for strict priors or use the apriori argument if supported.

# [TODO]: Predict probabilities on the training set
# nb_train_probs <- predict(nb_model, newdata = train_data, type = "raw")[, "Yes"]

# [TODO]: Threshold Tuning using ROC curve
# nb_roc <- roc(train_data$Revenue, nb_train_probs)
# optimal_nb_thresh <- coords(nb_roc, "best", ret = "threshold")$threshold

# [TODO]: Predict on test set, apply optimal threshold, and generate caret::confusionMatrix

cat("\n--- PART B: Linear Discriminant Analysis (LDA) ---\n")

# [TODO]: Train LDA using lda(). Include 'prior = balanced_priors'.
# lda_model <- lda(Revenue ~ ., data = train_data, prior = balanced_priors)

# [TODO]: Predict probabilities on train set, perform threshold tuning via ROC
# lda_train_probs <- predict(lda_model, newdata = train_data)$posterior[, "Yes"]

# [TODO]: Predict on test set, apply optimal threshold, and generate caret::confusionMatrix

# [TODO]: Plot ROC curves for both models  and save to outputs/figures/generative/

# 4. Save Models for Comparison Phase
# saveRDS(list(nb = nb_model, lda = lda_model), "outputs/models/generative_models.rds")

cat("\n--- Completed 04_generative.R ---\n")