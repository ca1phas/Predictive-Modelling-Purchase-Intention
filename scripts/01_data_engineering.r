# ==============================================================================
# Script: 01_data_engineering.r
# Tasks: 
# 1. Data Acquisition & Initial Cleaning (Remove duplicates, correct types).
# 2. Exploratory Data Analysis (Save distributions and boxplots to /figures).
# 3. Data Splitting (Chronological time-based split).
# 4. Feature Engineering (Probabilistic imputation, quarter binning, rare level bucketing).
# 5. Transformation & Scaling (One-Hot Encoding, log1p, Z-score standardization).
# 6. Exporting processed datasets and preprocessing artifacts for production.
# ==============================================================================

cat("\n--- Running 01_data_engineering.r ---\n")

# Load required libraries
library(ucimlrepo)
library(caret)

# Ensure output directories exist
if(!dir.exists("outputs/figures")) dir.create("outputs/figures", recursive = TRUE)
if(!dir.exists("outputs/data")) dir.create("outputs/data", recursive = TRUE)

cat("1. Fetching e-Commerce dataset (ID: 468) from UCI ML Repository...\n")
online_shoppers <- fetch_ucirepo(id = 468)
X <- online_shoppers$data$features
y <- online_shoppers$data$targets
df <- cbind(X, y)

cat("2. Understanding the dataset\n")
cat("Checking for duplicate rows...\n")
n_duplicates <- sum(duplicated(df))
cat(sprintf("Found %d duplicate rows.\n", n_duplicates))

# Remove duplicates, since they represent the exact copies of observations and do not add additional information
if(n_duplicates > 0) {
  df <- unique(df)
  cat("-> Successfully removed duplicate rows to reduce bias and overfitting\n")
}

# Convert categorical attributes to factors
cat_cols = c(
  "Month", 
  "OperatingSystems", 
  "Browser", 
  "Region", 
  "TrafficType", 
  "VisitorType",
  "Weekend",
  "Revenue"
)
df[cat_cols] = lapply(df[cat_cols], as.factor)

month_levels <- c("Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
df$Month <- factor(df$Month, levels = month_levels)

# Separate data into numerical and categorical attributes
num_df = df[sapply(df, is.numeric)]
cat_df = df[sapply(df, is.factor)]

# Numerical Data Summary
num_summary <- data.frame(
  Type = sapply(num_df, class),
  Missing_Values = sapply(num_df, function(x) sum(is.na(x))),
  Unique_Values = sapply(num_df, function(x) length(unique(x))),
  Min_Values = sapply(num_df, function(x) min(x, na.rm = TRUE)),
  Max_Values = sapply(num_df, function(x) max(x, na.rm = TRUE))
)

# Categorical Data Summary
cat_summary <- data.frame(
  Type = sapply(cat_df, class),
  Missing_Values = sapply(cat_df, function(x) sum(is.na(x))),
  Unique_Values = sapply(cat_df, function(x) length(unique(x))),
  Top_Category = sapply(cat_df, function(x) names(sort(table(x), decreasing=TRUE))[1])
)

cat("-- Preliminary Data Summary: Numerical Variables --\n")
print(num_summary)

cat("-- Preliminary Data Summary: Categorical Variables --\n")
print(cat_summary)

cat("3. Saving Histograms, Box Plots, Bar Plots as PNGs\n")

# Histograms for Numerical Attributes
png("outputs/figures/numerical_distributions.png", width=1600, height=800, res=150)
par(mfrow=c(2,5))
par(mar=c(6, 4, 3, 1))

for (col in names(num_df)) {
  hist(
    num_df[[col]],
    main=paste("Dist. of", col),
    xlab=col,
    col="skyblue",
    border="white"
  )
}
dev.off()

# Boxplots of Numerical Attributes
png("outputs/figures/numerical_boxplots.png", width=1600, height=800, res=150)
par(mfrow=c(2,5))
par(mar=c(6,4,3,1))

for (col in names(num_df)) {
  boxplot(
    num_df[[col]],
    main=paste("Boxplot of", col),
    ylab=col,
    col="lightgreen"
  )
}
dev.off()


# Barplots for Categorical Attributes
png("outputs/figures/categorical_distributions.png", width=1600, height=800, res=150)
par(mfrow=c(2,4))
par(mar=c(6,4,3,1))

for (col in names(cat_df)) {
  barplot(
    table(cat_df[[col]]),
    main=paste("Freq. of", col),
    col="lightcoral",
    las=2
  )
}
dev.off()
cat("All plots successfully saved to outputs/figures/\n")

cat("4. Data Splitting\n")
cat("Solution: Chronological Time-Based Split (Train = Jan-Oct, Test = Nov-Dec)\n")
cat("Why?\n
    1. Lecture guidelines: chronological split (t < tc) for time-dependent data.\n
    2. Test size is ~38%, similar to the recommended 30% threshold and 10-50% range recommended in the lecture notes.\n
    3. Reduce overfitting risk by evaluating models based on their ability 
    to  predict holiday-season (year-end) purchasing patterns based on regular-season (non-year-end) data\n")

# Split into Jan to oct & (Nov, Dec)
test_months <- c("Nov", "Dec")
train_data <- df[!df$Month %in% test_months, ]
test_data  <- df[df$Month %in% test_months, ]

cat(sprintf("-> Train Data: %d rows (%.1f%%)\n", nrow(train_data), (nrow(train_data)/nrow(df))*100))
cat(sprintf("-> Test Data: %d rows (%.1f%%)\n", nrow(test_data), (nrow(test_data)/nrow(df))*100))

cat("5. Data Cleaning & Feature Engineering\n")
cat("Based on the data summary, there is no empty columns.\n")

cat("Handle 'Other' Category for VisitorType when the uciml states that VisitorType is either returning or new visitor.\n")

cat("Solution: Probabilistic Imputation\n")
cat("Why?\n
    1. No Data Loss.\n
    2. Suitable for all models.\n
    3. Works for new incoming data with 'Other' category.\n")

# Calculate the number of Returning and New visitors
# USe train_data only to prevent data leakage
num_returning <- sum(train_data$VisitorType == "Returning_Visitor")
num_new <- sum(train_data$VisitorType == "New_Visitor")

# Calculate the ratio (probability of a known visitor being a Returning Visitor)
prob_returning <- num_returning / (num_returning + num_new)

# Create the new numeric feature 'isReturningVisitor'
train_data$isReturningVisitor <- ifelse(
  train_data$VisitorType == "Returning_Visitor", 1,
  ifelse(train_data$VisitorType == "New_Visitor", 0, prob_returning)
)
test_data$isReturningVisitor <- ifelse(
  test_data$VisitorType == "Returning_Visitor", 1,
  ifelse(test_data$VisitorType == "New_Visitor", 0, prob_returning)
)

# Drop the original categorical VisitorType column
train_data$VisitorType <- NULL
test_data$VisitorType <- NULL

cat("Converted VisitorType to numeric 'isReturningVisitor'.\n")
cat(sprintf("Assigned 'Other' the probabilistic value of: %.4f\n", prob_returning))

cat("Handle missing 'Jan' and 'Apr' Month data.\n")
cat("Solution: Feature Engineering (Quarter Binning)\n")
cat("Why?\n
    1. Prevents errors in Logistic Regression/LDA.\n
    2. Ensures no data loss.\n
    3. Makes the model deployment-ready for unseen Jan/Apr data.\n"
)

# Map existing and future months into 4 Quarters
train_data$Quarter <- ifelse(train_data$Month %in% c("Jan", "Feb", "Mar"), "Q1",
                             ifelse(train_data$Month %in% c("Apr", "May", "June"), "Q2",
                                    ifelse(train_data$Month %in% c("Jul", "Aug", "Sep"), "Q3", "Q4")))
train_data$Quarter <- as.factor(train_data$Quarter)


# Test_data only contains December, so it's guaranteed to be Q4
# The factor levels are matched to the training set to prevent pipeline bugs
test_data$Quarter <- factor(rep("Q4", nrow(test_data)), levels = levels(train_data$Quarter))
cat("-> Successfully grouped Months into Quarters (Q1, Q2, Q3, Q4).\n")

# Drop the 'Month' column so it doesn't break models during training
train_data$Month <- NULL
test_data$Month <- NULL

cat("All other categorical columns have at least one sample for each category.\n")

cat("Handle categorical levels with extremely low frequencies.\n")
cat("Solution: Combine these rare levels into an 'Other' bucket.\n")
cat("Why?\n
    1. Prevents generating dozens of Near-Zero Variance columns during One-Hot Encoding.\n
    2. Prevents singular matrix error in LDA  and Logistic Regression.\n
    3. Retains the data rather than blindly deleting the columns.\n")

cols_to_lump <- c("OperatingSystems", "Browser", "Region", "TrafficType")
threshold <- 0.05 # 5% threshold

production_keep_levels <- list() # For production artifacts

for (col in cols_to_lump) {
  # 1. Calculate frequencies using ONLY training data to prevent data leakage
  freqs <- prop.table(table(train_data[[col]]))
  
  # 2. Identify levels that appear in at least [threshold x 100]% of the training data
  keep_levels <- names(freqs[freqs >= threshold])
  production_keep_levels[[col]] <- keep_levels
  
  # 3. Modify training data: Keep common levels, replace rest with 'Other'
  train_levels <- as.character(train_data[[col]])
  train_levels[!train_levels %in% keep_levels] <- "Other"
  train_data[[col]] <- as.factor(train_levels)
  
  # 4. Modify testing data using the EXACT same rules learned from training
  test_levels <- as.character(test_data[[col]])
  test_levels[!test_levels %in% keep_levels] <- "Other"
  # Match factor levels precisely to training data
  test_data[[col]] <- factor(test_levels, levels = levels(train_data[[col]]))
}

cat("-> Successfully combine rare levels in OperatingSystems, Browser, Region, and TrafficType into 'Other'.\n")

cat("6. One-Hot Encoding\n")
cat("Handle all nominal categorical predictors.\n")
cat("Solution: One-Hot Encoding using caret::dummyVars with fullRank=TRUE.\n")
cat("Why?\n
    1. Prevents models from misinterpreting nominal categories as mathematical ranks.\n
    2. fullRank=TRUE avoids the dummy variable trap (perfect multicollinearity).\n")

oneh_model <- dummyVars( ~ OperatingSystems + Browser + Region + 
                           TrafficType + Weekend + Quarter, data=train_data, fullRank=TRUE)
train_ohe <- predict(oneh_model, newdata=train_data)
test_ohe <- predict(oneh_model, newdata=test_data)

train_data <- cbind(train_data, as.data.frame(train_ohe))
test_data <- cbind(test_data, as.data.frame(test_ohe))

# Clean up by removing the original categorical columns
cols_to_remove <- c("OperatingSystems", "Browser", "Region", "TrafficType", "Weekend", "Quarter")
train_data[cols_to_remove] <- list(NULL)
test_data[cols_to_remove] <- list(NULL)
cat("-> Successfully applied One-Hot Encoding to OperatingSystems, Browser, Region, TrafficType, Weekend, and Quarter.\n")

cat("7. Handling class imbalance\n")

# The training proportion
prop.table(table(train_data$Revenue)) # 87.6:12.4
table(train_data$Revenue) # n_minority=932
length(names(train_data)) -1  # no_features=32

cat("Since the imbalance (ratio ~7.06, n_minority=932, no_features=32) is moderately imbalance,\n")
cat("it's best to fix imbalance as close to the source as possible within each model's learning method.\n\n")

cat("Logistic Regression & Decision Tree -> use weights + threshold tuning\n")
cat("Weighted kNN -> SMOTE (inside CV Loop to prevent data leakage)\n\n")

cat("Generative Models (Naive Bayes, LDA) -> priors = c(0.5, 0.5) + threshold tuning\n")

cat("Random Forest -> classwt/sampsize + threshold tuning\n")
cat("XGBoost -> scale_pos_weight = 5.25+ eval_metric = \"aucpr\"\n")
cat("LightGBM -> is_unbalance = TRUE + metric = \"averagae_precision\"\n")
cat("CatBoost -> auto_class_weights = \"Balanced\" + eval_metric = \"PRAUC\"\n")

cat("8. Feature Transformation & Scaling\n")
cat("Handle severe positive skewness, magnitude differences and outliers, in numerical features.\n")
cat("Solution: Log-Plus-One transformation followed by Standard Scaling (Z-score).\n")
cat("Why?\n
  1. log1p() fixes right-skewness while handling the value 0.\n
  2. Normalizing distributions improves the performances of non-tree based models\n
  3. Easily reversable for interpretability.\n
")

num_cols = names(num_df)

# 8.1. Log-Plus-One Transformation (since all numerical mins =- 0)
train_data[num_cols] <- lapply(train_data[num_cols], log1p)
test_data[num_cols] <- lapply(test_data[num_cols], log1p)
cat("-> Successfully applied log1p() transformation to numerical features.\n")

# 8.2. Standardization
train_scaled_obj <- scale(train_data[num_cols])
train_means <- attr(train_scaled_obj, "scaled:center")
train_sds   <- attr(train_scaled_obj, "scaled:scale")

train_data[num_cols] <- as.data.frame(train_scaled_obj)

# Use training means & s.d.s to prevent data leakage
test_data[num_cols] <- as.data.frame(scale(test_data[num_cols], center = train_means, scale = train_sds))

cat("-> Successfully applied Standardization.\n")

# 8.3 Formatting Target Variable for caret
# The values TRUE/FALSE are R's reserved logical keywords
# Using them as factor levels causes R unable to distinguish the boolean value TRUE and a factor level named TRUE
# Some of caret's calculation may cause require the target factor levels to be valid R variable names.
# Need to map TRUE/FALSE in Revenue to "Yes"/"No"

# Convert TRUE/FALSE to YES/NO to prevent caret's compilation errors
train_data$Revenue <- factor(ifelse(train_data$Revenue == TRUE, "Yes", "No"), levels = c("No", "Yes"))
test_data$Revenue <- factor(ifelse(test_data$Revenue == TRUE, "Yes", "No"), levels = c("No", "Yes"))
cat("-> Succesfully formatted Revenue factor levels to be valid R variable names to prevent caret's compilation error.\n")

cat("8.4 Remove Near-Zero Variance (NZV) Predictors\n")

# Identify features with near-zero variance
nzv_cols <- nearZeroVar(train_data, saveMetrics = TRUE)
nzv_feature_names <- rownames(nzv_cols[nzv_cols$nzv == TRUE, ])

if(length(nzv_feature_names) > 0) {
  cat("Warning: The following features have near-zero variance:\n")
  print(nzv_feature_names)
  cat("\n*** PIPELINE NOTE FOR MODELERS ***\n")
  cat("-> These features have NOT been removed from the global training set.\n")
  cat("-> For LogReg/LDA: You might need to manually drop these columns in your individual modeling scripts to prevent perfect separation or singularity errors.\n")
} else {
  cat("-> No near-zero variance features detected.\n")
  train_data_parametric <- train_data
  test_data_parametric <- test_data
}

cat("9. Saving training and testing datasets\n")

saveRDS(train_data, "outputs/data/train_data.rds")
saveRDS(test_data, "outputs/data/test_data.rds")

cat("Successfully saved training and testing datasets as .rds files in outputs/data.\n")

cat("10. Saving preprocessing artifacts")
preprocessing_artifacts <- list(
  prob_returning = prob_returning,
  production_keep_levels = production_keep_levels,
  oneh_model = oneh_model,
  train_means = train_means,
  train_sds = train_sds
)
saveRDS(preprocessing_artifacts, "outputs/data/preprocessing_artifacts.rds")
cat("-> Successfully saved preprocessing artifacts for production deployment.\n")

cat ("\n--- Completed 01_data_engineering.r ---\n")