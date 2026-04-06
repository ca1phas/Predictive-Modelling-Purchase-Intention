# Run this to update renv.lock with the same packages versions
# Only use when you added a new library to the code
# renv::snapshot()


# ==============================================================================
# scripts/00_main.R
# Main execution pipeline for UECM3993 Predictive Modelling Assignment
# ==============================================================================

# 1. Environment Setup
# Ensure renv is installed and restore the project's exact package versions
if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv")
renv::restore()

# 2. Pipeline Execution
# Note: Assuming working directory is the project root (where the .Rproj file is)
scripts_dir <- "scripts/"

cat("Starting Predictive Modelling Pipeline...\n")

cat("\n--- Phase 1: Data Engineering ---\n")
source(file.path(scripts_dir, "01_data_engineering.r"))

cat("\n--- Phase 2: Unsupervised Modelling ---\n")
source(file.path(scripts_dir, "02_unsupervised.R"))

cat("\n--- Phase 3: Discriminative Models ---\n")
source(file.path(scripts_dir, "03_discriminative.R"))
source(file.path(scripts_dir, "03_discriminative_Ridge_Lasso_ElasticNet.R"))

cat("\n--- Phase 4: Generative Models & Feature Selection ---\n")
source(file.path(scripts_dir, "04_generative.R"))

cat("\n--- Phase 5: Advanced/Flexible Models ---\n")
source(file.path(scripts_dir, "05_ensemble.R"))

cat("\n--- Phase 6: Model Comparison ---\n")
source(file.path(scripts_dir, "06_comparison.R"))

cat("\nPipeline Execution Complete. All outputs saved to /outputs.\n")