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

# Calculate sample sizes for balanced Random Forest (Downsampling the majority class to 1:1)
n_minority <- sum(train_data$Revenue == "Yes")
rf_sampsize <- c("No" = n_minority, "Yes" = n_minority)

# Prepare DMatrix for XGBoost (Requires numeric matrices)
X_train_mat <- model.matrix(Revenue ~ . - 1, data = train_data)
y_train_num <- ifelse(train_data$Revenue == "Yes", 1, 0)

X_test_mat <- model.matrix(Revenue ~ . - 1, data = test_data)
y_test_num <- ifelse(test_data$Revenue == "Yes", 1, 0)

# Function to evaluate classification metrics
evaluate_model <- function(model, data){
  pred <- model$predicted
  actual <- data$Revenue
  cm <- confusionMatrix(pred, actual, positive = "Yes")
  precision <- cm$byClass["Precision"]
  recall <- cm$byClass["Recall"]
  f1 <- cm$byClass["F1"]
  bal_acc <- cm$byClass["Balanced Accuracy"]
  return(c(precision, recall, f1, bal_acc))
}

# ==============================================================================
# PART A: RANDOM FOREST
# ==============================================================================

cat("\n--- PART A: Random Forest (Bagging) ---\n")

# =========================
# Helper Functions
# =========================
# Evaluate classification metrics
evaluate_model <- function(model, data){
  cm <- caret::confusionMatrix(model$predicted, data$Revenue, positive="Yes")
  c(Precision=cm$byClass["Precision"],
    Recall=cm$byClass["Recall"],
    F1=cm$byClass["F1"],
    Balanced_Accuracy=cm$byClass["Balanced Accuracy"])
}

# Train RF with flexible parameters
train_rf <- function(ntree, mtry=NULL, nodesize=NULL, sampsize=NULL){
  
  # Create base arguments and add optional parameters only if provided
  args <- list(formula = Revenue ~ .,data = train_data, ntree = ntree)
  
  if(!is.null(mtry)) args$mtry <- mtry
  if(!is.null(nodesize)) args$nodesize <- nodesize
  if(!is.null(sampsize)) args$sampsize <- sampsize
  
  do.call(randomForest, args)
}

# Extract final OOB error
get_oob <- function(model) tail(model$err.rate[,"OOB"],1)

# =========================
# 1. Default Model
# =========================
cat("1: Training default Random Forest...\n")

def_mtry <- floor(sqrt(ncol(train_data)-1))
def_ntree <- 500

rf_model <- randomForest(
  Revenue~., data=train_data,
  mtry=def_mtry, sampsize=rf_sampsize,
  ntree=def_ntree, importance=TRUE
)

default_metric <- evaluate_model(rf_model, train_data)
cat("Model trained with default parameter with 1:1 sampling ratio\n")
print(default_metric)

# =========================
# 2. Find Stable ntree
# =========================
cat("2: Assessing tree stability...\n")

set.seed(42)
test_tree <- 800; err_threshold <- 0.0005

stability_rf <- train_rf(ntree=test_tree)
oob_error <- stability_rf$err.rate[,"OOB"]
error_diff <- abs(diff(oob_error))

stable_ntree <- test_tree; count <- 0
for(i in seq_along(error_diff)){
  count <- ifelse(error_diff[i] < err_threshold, count+1,0)
  if(count>=50){ stable_ntree <- i+1; break }
}

cat(sprintf("Stable ntree: %d\n", stable_ntree))

# Stability plot
png("outputs/figures/ensemble/rf_stability_plot.png",800,600)
plot(oob_error,type="l",lwd=2,xlab="Trees",ylab="OOB Error",
     main="RF Tree Stability")
abline(v=stable_ntree,col="blue",lty=2)
legend("topright", legend=paste("Stable ntree =", stable_ntree), 
       col="blue", lty=2, lwd=1)
dev.off()

# =========================
# Sensitivity Table
# =========================

sensitivity_results <- data.frame(
  Parameter=c("ntree","mtry","nodesize","sampsize","Refinement"),
  Selected_Value=NA,Best_OOB_Error=NA,
  Precision=NA,Recall=NA,F1=NA,Balanced_Accuracy=NA
)

# -------------------------
# Baseline (ntree)
# -------------------------

rf_base <- train_rf(ntree=stable_ntree)
metrics <- evaluate_model(rf_base,train_data)

sensitivity_results[1,2:7] <- c(stable_ntree,get_oob(rf_base),metrics)

# =========================
# 3. Tune mtry
# =========================
cat("Perform parameters tuning...This may take a while\n")
cat("The optimal parameter will be shown once all are found\n")
cat("3: Tuning mtry...\n")

mtry_grid <- seq(2,floor(ncol(train_data)/2),by=2)

oob_mtry <- sapply(mtry_grid,function(m)
  get_oob(train_rf(ntree=stable_ntree,mtry=m)))

best_mtry <- mtry_grid[which.min(oob_mtry)]

rf_mtry <- train_rf(ntree=stable_ntree,mtry=best_mtry)
metrics <- evaluate_model(rf_mtry,train_data)

sensitivity_results[2,2:7] <- c(best_mtry,min(oob_mtry),metrics)
cat("mtry tuning completed\n")

# =========================
# 4. Tune nodesize
# =========================
cat("4: Tuning nodesize...\n")

node_grid <- c(1,5,10,20,50)

oob_node <- sapply(node_grid,function(n)
  get_oob(train_rf(ntree=stable_ntree,mtry=best_mtry,nodesize=n)))

best_nodesize <- node_grid[which.min(oob_node)]

rf_node <- train_rf(ntree=stable_ntree,mtry=best_mtry,nodesize=best_nodesize)
metrics <- evaluate_model(rf_node,train_data)

sensitivity_results[3,2:7] <- c(best_nodesize,min(oob_node),metrics)
cat("nodesize tuning completed\n")

# =========================
# 5. Tune sampsize
# =========================
cat("5: Tuning sampsize...\n")

n_min <- sum(train_data$Revenue=="Yes")
ratios <- c(1,1.5,2)

oob_sample <- sapply(ratios,function(r){
  sz <- c("No"=as.integer(n_min*r),"Yes"=n_min)
  get_oob(train_rf(ntree=stable_ntree,
                   mtry=best_mtry,
                   nodesize=best_nodesize,
                   sampsize=sz))
})

best_ratio <- ratios[which.min(oob_sample)]

final_sz <- c("No"=as.integer(n_min*best_ratio),"Yes"=n_min)

rf_sample <- train_rf(ntree=stable_ntree,
                      mtry=best_mtry,
                      nodesize=best_nodesize,
                      sampsize=final_sz)

metrics <- evaluate_model(rf_sample,train_data)

sensitivity_results[4,2:7] <- c(paste0("1:",best_ratio),min(oob_sample),metrics)
cat("sampsize tuning completed\n")

# =========================
# 6. Refinement (re-check mtry)
# =========================
cat("6: Refinement...\n")

oob_refine <- sapply(mtry_grid,function(m)
  get_oob(train_rf(ntree=stable_ntree,
                   mtry=m,
                   nodesize=best_nodesize,
                   sampsize=final_sz)))

best_mtry <- mtry_grid[which.min(oob_refine)]

rf_final <- train_rf(ntree=stable_ntree,
                     mtry=best_mtry,
                     nodesize=best_nodesize,
                     sampsize=final_sz)

metrics <- evaluate_model(rf_final,train_data)

sensitivity_results[5,2:7] <- c(best_mtry,min(oob_refine),metrics)
cat("Sensitvity results:\n")
print(sensitivity_results)

# =========================
# 7. Train Tuned Model
# =========================
cat("7: Training tuned model...\n")

rf_model_tune <- randomForest(
  Revenue~.,data=train_data,
  mtry=best_mtry,nodesize=best_nodesize,
  sampsize=final_sz,ntree=stable_ntree,
  importance=TRUE)

tuned_metric <- evaluate_model(rf_model_tune,train_data)
cat("Tuned model performance on train data\n")
print(tuned_metric)

# =========================
# 8. Model Comparison
# =========================
cat("8: Comparing default vs tuned model...\n")

test_probs_def <- predict(rf_model,test_data,type="prob")[,"Yes"]
test_probs_tune <- predict(rf_model_tune,test_data,type="prob")[,"Yes"]

# We use the default threshold (i.e. 0.5)
test_pred_def <- factor(ifelse(test_probs_def>=0.5,"Yes","No"),levels=c("No","Yes"))
test_pred_tune <- factor(ifelse(test_probs_tune>=0.5,"Yes","No"),levels=c("No","Yes"))

cat("Default Model:\n")
print(confusionMatrix(test_pred_def,test_data$Revenue,positive="Yes"))

cat("Tuned Model:\n")
print(confusionMatrix(test_pred_tune,test_data$Revenue,positive="Yes"))

# =========================
# 9. Threshold Optimization (Model with Default Parameter is chosen as winner)
# =========================
cat("9: Threshold tuning...\n")

prob_oob <- rf_model$votes[,"Yes"]
thresholds <- seq(0.2,0.6,by=0.02)

bal_acc <- sapply(thresholds,function(t){
  pred <- factor(ifelse(prob_oob>t,"Yes","No"),levels=c("No","Yes"))
  confusionMatrix(pred,train_data$Revenue,positive="Yes")$byClass["Balanced Accuracy"]
})

best_threshold <- thresholds[which.max(bal_acc)]

png("outputs/figures/ensemble/rf_threshold_curve.png", width=800, height=600)
plot(thresholds, bal_acc, type="l", lwd=2,
     xlab="Threshold", ylab="Balanced Accuracy")
dev.off()
cat("Threshold tuning completed the final output will be shown at the end of random forest section\n")

# =========================
# 10. Feature Importance
# =========================
cat("10: Feature importance...\n")
cat("Generating feature importance image\n")
png("outputs/figures/ensemble/rf_var_imp.png",800,600)
varImpPlot(rf_model,main="Feature Importance: Purchase Intention")
dev.off()
cat("Image generating done at outputs/figures/ensemble/rf_var_imp.png\n")

# =========================
# 11. Partial Dependence Plots
# =========================
cat("11: PDP analysis...\n")

imp_matrix <- importance(rf_model)
top_features <- rownames(imp_matrix)[order(imp_matrix[,4],decreasing=TRUE)[1:3]]

png("outputs/figures/ensemble/rf_pdp_plots.png",1000,500)
par(mfrow=c(1,3))
for(i in 1:length(top_features))
  partialPlot(rf_model,train_data,top_features[i],which.class="Yes",
              main=paste("PDP:",top_features[i]),
              xlab=f,ylab="Logit Prob of Revenue")
dev.off()
cat("PDP analysis done, related image is generated at outputs/figures/ensemble/rf_pdp_plots.png\n")

# =========================
# 12. Final Model Evaluation
# =========================
cat("Final model evaluation with optimized threshold...\n")

prob_test <- predict(rf_model, test_data, type="prob")[,"Yes"]

final_pred <- factor(
  ifelse(prob_test > best_threshold,"Yes","No"),
  levels=c("No","Yes")
)

final_cm <- confusionMatrix(final_pred, test_data$Revenue, positive="Yes")

print(final_cm)

# =========================
# 13. Save Final Model
# =========================
cat("Save the final paremter for handing off\n")
saveRDS(list(
  model=rf_model,
  predicted_probs=test_probs_def,
  threshold=best_threshold,
  best_params=list(mtry=def_mtry,sampsize=rf_sampsize,ntree=def_ntree)
),"outputs/models/rf_final_packet.rds")

cat("\n--- PART A: Random Forest Completed ---\n")



# ==============================================================================
# PART B: GRADIENT BOOSTING
# ==============================================================================

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