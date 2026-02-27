# ==============================================================================
# Script: 06_comparison.R
# Author: Casimir Chiong Ming Yuan (2401315)
# Tasks: 
# 1. Load the preserved test_data.rds
# 2. Load all saved supervised models from outputs/models/
# 3. Generate predictions and extract metrics (Accuracy, Kappa, AUC, etc.)
# 4. Compile the final "Model Comparison" table
# ==============================================================================

cat("\n--- Running 06_comparison.R ---\n")

# 1. Load Required Libraries
library(caret)       # For Confusion Matrix and Statistics
library(pROC)        # For AUC calculations

# Ensure output directory exists for final reports/tables
if(!dir.exists("outputs/tables")) dir.create("outputs/tables", recursive = TRUE)

# 2. Load the Preserved Testing Data
cat("Loading test data...\n")
test_data <- readRDS("outputs/data/test_data.rds")
y_test <- test_data$Revenue
y_test_num <- ifelse(y_test == "Yes", 1, 0) # Numeric version for ROC/AUC

# Prepare DMatrix for XGBoost (if Kok Shau Loon uses it)
library(xgboost)
X_test_mat <- model.matrix(Revenue ~ . - 1, data = test_data)

# 3. Load All Saved Models safely
cat("Loading trained models...\n")

# Initialize an empty list to store loaded models
models <- list()

# Helper function to load models if the file exists
load_if_exists <- function(filepath) {
  if(file.exists(filepath)) {
    return(readRDS(filepath))
  } else {
    cat(sprintf("Warning: %s not found. Skipping...\n", filepath))
    return(NULL)
  }
}

disc_models <- load_if_exists("outputs/models/discriminative_models.rds")
gen_models  <- load_if_exists("outputs/models/generative_models.rds")
ens_models  <- load_if_exists("outputs/models/ensemble_models.rds")

# Combine all available models into one flat list
if(!is.null(disc_models)) models <- c(models, disc_models)
if(!is.null(gen_models))  models <- c(models, gen_models)
if(!is.null(ens_models))  models <- c(models, ens_models)

# 4. Evaluation Loop
cat("Evaluating models and extracting metrics...\n")

# Initialize an empty dataframe to hold the results
comparison_table <- data.frame(
  Model = character(),
  Accuracy = numeric(),
  Kappa = numeric(),
  Sensitivity = numeric(),
  Specificity = numeric(),
  AUC = numeric(),
  stringsAsFactors = FALSE
)

# Note: In a real run, you will need to adjust the predict() syntax based on 
# the specific model type (e.g., type = "response" for glm, type = "raw" for naiveBayes).
# This is a generalized structure Casimir will need to adapt based on his team's exact output.

# [TODO]: Loop through `models` (or evaluate them one by one)
# [TODO]: Generate predictions using predict()
# [TODO]: Apply the specific optimal thresholds tuned by each modeler
# [TODO]: Generate caret::confusionMatrix(predicted_classes, y_test, positive = "Yes")
# [TODO]: Calculate AUC using pROC::roc(y_test_num, predicted_probabilities)

# Example of appending a row (Casimir will fill this with actual variables):
# comparison_table <- rbind(comparison_table, data.frame(
#   Model = "Logistic Regression (Ridge)",
#   Accuracy = round(cm$overall["Accuracy"], 4),
#   Kappa = round(cm$overall["Kappa"], 4),
#   Sensitivity = round(cm$byClass["Sensitivity"], 4),
#   Specificity = round(cm$byClass["Specificity"], 4),
#   AUC = round(roc_obj$auc, 4)
# ))

# 5. Display and Save Final Table
cat("\n=======================================================\n")
cat("                 FINAL MODEL COMPARISON                  \n")
cat("=======================================================\n")

# Sort table by AUC (or your preferred metric) descending
# comparison_table <- comparison_table[order(-comparison_table$AUC), ]

# Print nicely in console
# print(kable(comparison_table, format = "markdown"))

# Save to CSV for the final report
# write.csv(comparison_table, "outputs/tables/final_model_comparison.csv", row.names = FALSE)
# cat("\n-> Saved comparison table to outputs/tables/final_model_comparison.csv\n")

cat("\n--- Completed 06_comparison.R ---\n")