cat("\nComparing Ridge vs Lasso vs Elastic Net...\n")

alphas <- c(0, 0.5, 1)
model_names <- c("Ridge", "ElasticNet", "Lasso")
results_list <- list()

set.seed(123)

for (i in seq_along(alphas)) {
  
  alpha_val <- alphas[i]
  name <- model_names[i]
  
  cat(sprintf("\nTraining %s (alpha = %.1f)...\n", name, alpha_val))
  
  model <- cv.glmnet(
    X_train, y_train, family = "binomial", alpha = alpha_val, weights = weights
  )
  
  # Train probabilities
  compare_train_probs <- predict(model, newx = X_train, type = "response", s = "lambda.min")
  
  # ROC + threshold tuning
  compare_roc <- roc(y_train, as.numeric(compare_train_probs))
  optimal_compare_thresh <- coords(compare_roc, "best", ret = "threshold")$threshold
  
  # Test probabilities
  compare_test_probs <- predict(model, newx = X_test, type = "response", s = "lambda.min")
  
  compare_pred <- ifelse(compare_test_probs > optimal_compare_thresh, "Yes", "No")
  compare_pred <- factor(compare_pred, levels = c("No", "Yes"))
  
  compare_cm <- confusionMatrix(compare_pred, y_test)
  compare_auc <- auc(roc(y_test, as.numeric(compare_test_probs)))
  
  results_list[[name]] <- data.frame(
    Model = name,
    Alpha = alpha_val,
    Accuracy = compare_cm$overall["Accuracy"],
    Kappa = compare_cm$overall["Kappa"],
    AUC = as.numeric(compare_auc)
  )
}

# Combine results
comparison_table <- do.call(rbind, results_list)
print(comparison_table)
