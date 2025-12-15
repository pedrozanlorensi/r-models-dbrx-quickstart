# R Models on Databricks — Technical Instructions

This document provides comprehensive technical instructions for training, logging, registering, and deploying R models on Databricks using MLflow and Unity Catalog.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Environment Setup](#environment-setup)
4. [Training R Models](#training-r-models)
5. [MLflow Integration](#mlflow-integration)
6. [Unity Catalog Registration](#unity-catalog-registration)
7. [Running Inferences](#running-inferences)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Overview

This guide covers the end-to-end workflow for productionizing R models on Databricks:

- **Train** R models using native R libraries (`rpart`, `caret`, etc.)
- **Log** models and metrics to MLflow for experiment tracking
- **Register** models to Unity Catalog for governance and versioning
- **Deploy** models for batch inference using the registered model

### Key Technologies

| Component | Purpose |
|-----------|---------|
| **sparklyr** | R interface to Apache Spark for distributed data processing |
| **MLflow R** | Experiment tracking, model logging, and artifact management |
| **carrier** | Model serialization for portable R functions |
| **Unity Catalog** | Centralized governance, versioning, and model registry |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING WORKFLOW                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ Unity Catalog│───▶│   R Model    │───▶│  MLflow Tracking     │  │
│  │   (Data)     │    │   Training   │    │  (Experiments/Runs)  │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                    │                │
│                                                    ▼                │
│                                          ┌──────────────────────┐  │
│                                          │   Unity Catalog      │  │
│                                          │   (Model Registry)   │  │
│                                          └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       INFERENCE WORKFLOW                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Unity Catalog      │───▶│   R Model    │───▶│ Unity Catalog│  │
│  │   (Model Registry)   │    │  Inference   │    │  (Results)   │  │
│  └──────────────────────┘    └──────────────┘    └──────────────┘  │
│                                     ▲                               │
│                                     │                               │
│                            ┌──────────────┐                         │
│                            │ Unity Catalog│                         │
│                            │ (Input Data) │                         │
│                            └──────────────┘                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Environment Setup

### Required R Packages

| Package | Purpose | Installation |
|---------|---------|--------------|
| `sparklyr` | Spark connectivity | Pre-installed on Databricks |
| `dplyr` | Data manipulation | Pre-installed on Databricks |
| `mlflow` | Experiment tracking | Pre-installed on Databricks |
| `rpart` | Decision tree models | Pre-installed on Databricks |
| `carrier` | Model serialization | **Requires installation** |

### Installing Packages

```r
# Required for model serialization
install.packages("carrier")

# If mlflow needs updating (rare)
install.packages("mlflow")
```

### Python Environment (for UC Registration)

The MLflow Python client is required for Unity Catalog model registration:

```python
%pip install --upgrade "mlflow[databricks]==3.5.0"
dbutils.library.restartPython()
```

---

## Training R Models

### Step 1: Connect to Spark

```r
library(sparklyr)
library(dplyr)

# Establish Spark connection via Databricks integration
spark <- spark_connect(method = "databricks")
```

### Step 2: Load Data from Unity Catalog

```r
# Reference a table in Unity Catalog
iris_tbl <- tbl(spark, "catalog_name.schema_name.table_name")

# Collect to local R dataframe for modeling
# Use for datasets that fit in memory
iris_df <- collect(iris_tbl)
```

> **Important:** For large datasets, consider using Spark-based ML via sparklyr's `ml_*` functions instead of collecting to R.

### Step 3: Data Preparation

```r
# Convert target variable to factor for classification
iris_df$species <- as.factor(iris_df$species)

# Train/test split
set.seed(42)  # For reproducibility
train_idx <- sample(seq_len(nrow(iris_df)), size = 0.8 * nrow(iris_df))
train_df <- iris_df[train_idx, ]
test_df <- iris_df[-train_idx, ]
```

### Step 4: Train the Model

```r
library(rpart)

# Train a decision tree classifier
model <- rpart(
  species ~ sepal_length_cm + sepal_width_cm + petal_length_cm + petal_width_cm,
  data = train_df,
  method = "class"  # Use "anova" for regression
)
```

### Step 5: Evaluate the Model

```r
# Predict on test set
predictions <- predict(model, test_df, type = "class")

# Calculate accuracy
accuracy <- mean(predictions == test_df$species)
cat(sprintf("Test accuracy: %.3f\n", accuracy))
```

---

## MLflow Integration

### Setting Up Experiments

```r
library(mlflow)

# Set experiment path (user-scoped or shared)
mlflow_set_experiment("/Users/your_email@databricks.com/my_r_experiment")
```

### Defining Model Signatures

Model signatures are **required** for Unity Catalog registration. They define the input/output schema:

```r
signature <- list(
  inputs = list(
    list(type = "double", name = "sepal_length"),
    list(type = "double", name = "sepal_width"),
    list(type = "double", name = "petal_length"),
    list(type = "double", name = "petal_width")
  ),
  outputs = list(
    list(type = "string")  # For classification
    # Use list(type = "double") for regression
  )
)
```

#### Supported Types

| Type | Use Case |
|------|----------|
| `double` | Numeric features, regression outputs |
| `float` | Lower-precision numerics |
| `integer` | Discrete numeric values |
| `long` | Large integers |
| `string` | Categorical features, classification outputs |
| `boolean` | Binary flags |

### Patching `mlflow_log_model` for Signature Support

The R MLflow client requires a patch to support signatures for Unity Catalog:

```r
mlflow_log_model <- function(model, artifact_path, signature = NULL, ...) {
  format_signature <- function(signature) {
    lapply(signature, function(x) {
      jsonlite::toJSON(x, auto_unbox = TRUE)
    })
  }
  temp_path <- fs::path_temp(artifact_path)
  model_spec <- mlflow_save_model(
    model, path = temp_path, model_spec = list(
      utc_time_created = mlflow:::mlflow_timestamp(),
      run_id = mlflow:::mlflow_get_active_run_id_or_start_run(),
      artifact_path = artifact_path, 
      flavors = list(),
      signature = format_signature(signature)
    ), ...
  )
  res <- mlflow_log_artifact(path = temp_path, artifact_path = artifact_path)
  tryCatch({
    mlflow:::mlflow_record_logged_model(model_spec)
  }, error = function(e) {
    warning(
      paste("Logging model metadata to the tracking server has failed.",
            "The model artifacts have been logged successfully.")
    )
  })
  res
}

# Override in namespace
assignInNamespace("mlflow_log_model", mlflow_log_model, ns = "mlflow")
```

> **Reference:** [Log R Models to Unity Catalog](https://zacdav-db.github.io/dbrx-r-compendium/chapters/mlflow/log-to-uc.html)

### Wrapping Models with Carrier

The `carrier` package creates portable function objects ("crates") that bundle code and dependencies:

```r
library(carrier)

# Create a crate that bundles the predict function with the model
r_func <- carrier::crate(
  function(newdata) {
    stats::predict(model, newdata = newdata, type = "class")
  },
  model = model  # Bundle the trained model
)
```

### Logging to MLflow

```r
# Start a new run
run <- mlflow_start_run()

# Log metrics
mlflow_log_metric("accuracy", accuracy)
mlflow_log_metric("num_features", 4)

# Log parameters
mlflow_log_param("model_type", "decision_tree")
mlflow_log_param("train_size", nrow(train_df))

# Log the model with signature
mlflow_log_model(r_func, "iris_r_class_model", signature = signature)

# End the run
mlflow_end_run()

# Retrieve run ID for registration
cat("Run ID:", run$run_id, "\n")
```

---

## Unity Catalog Registration

### Why Register to Unity Catalog?

- **Governance:** Centralized access control and auditing
- **Versioning:** Automatic version tracking for each registered model
- **Aliases:** Use semantic aliases like `champion`, `challenger`, `production`
- **Lineage:** Track data and model dependencies

### Registration Process (Python Required)

Registration must be done via the Python MLflow client:

```python
%python
import mlflow
from mlflow.tracking import MlflowClient

# Configure Unity Catalog as registry
mlflow.set_registry_uri("databricks-uc")

# Build model URI from run
run_id = "your_run_id_here"
artifact_path = "iris_r_class_model"
run_uri = f"runs:/{run_id}/{artifact_path}"

# Full model name in Unity Catalog format
catalog = "my_catalog"
schema = "my_schema"
model_name = "iris_r_class_model_uc"
full_model_name = f"{catalog}.{schema}.{model_name}"

# Register the model
registered_model = mlflow.register_model(run_uri, full_model_name)

print(f"Registered version: {registered_model.version}")
```

### Setting Model Aliases

```python
%python
client = MlflowClient(registry_uri="databricks-uc")

# Set alias for deployment
client.set_registered_model_alias(
    name=full_model_name,
    alias="champion",      # or "production", "staging", etc.
    version=registered_model.version
)

# Delete an alias
client.delete_registered_model_alias(name=full_model_name, alias="old_alias")
```

### Model URI Formats

| Format | Example |
|--------|---------|
| By Version | `models:/catalog.schema.model/1` |
| By Alias | `models:/catalog.schema.model@champion` |
| By Run | `runs:/run_id/artifact_path` |

---

## Running Inferences

### Loading Registered Models

```r
library(mlflow)
library(carrier)

# Load model using alias (recommended)
model <- mlflow_load_model("models:/my_catalog.my_schema.iris_r_model_uc@champion")

# Or load specific version
model <- mlflow_load_model("models:/my_catalog.my_schema.iris_r_model_uc/3")
```

### Batch Inference Pattern

```r
library(sparklyr)
library(dplyr)

# Connect to Spark
spark <- spark_connect(method = "databricks")

# Load input data
input_tbl <- tbl(spark, "my_catalog.my_schema.input_features")
input_df <- collect(input_tbl)

# Run predictions (model is a crate function)
predictions <- model(input_df)

# Add predictions to dataframe
input_df$prediction <- predictions

# Copy results to Spark
results_tbl <- copy_to(spark, input_df, "results_temp", overwrite = TRUE)

# Write to Unity Catalog
spark_write_table(
  results_tbl,
  name = "my_catalog.my_schema.predictions_output",
  mode = "overwrite"  # or "append"
)
```

### Inference Input Requirements

The input dataframe must have columns matching the model signature:

```r
# Check expected inputs
print(model)

# Ensure columns exist and have correct types
input_df <- input_df %>%
  select(
    sepal_length = sepal_length_cm,  # Rename if needed
    sepal_width = sepal_width_cm,
    petal_length = petal_length_cm,
    petal_width = petal_width_cm
  )
```

---

## Best Practices

### Model Development

1. **Use reproducible seeds:** Always set `set.seed()` before random operations
2. **Document features:** Include feature engineering steps in notebooks
3. **Version control:** Store notebooks in Git-enabled repos
4. **Log everything:** Log hyperparameters, metrics, and artifacts to MLflow

### Model Signatures

1. **Always define signatures:** Required for Unity Catalog, helpful for documentation
2. **Match column names exactly:** Signature names must match inference input columns
3. **Use appropriate types:** Match R types to signature types carefully

### Unity Catalog

1. **Use meaningful model names:** Follow naming conventions (`project_model_type`)
2. **Leverage aliases:** Use `champion`/`challenger` pattern for A/B testing
3. **Add descriptions:** Document models in the Unity Catalog UI
4. **Set permissions:** Configure access control for models

### Production Deployment

1. **Pin versions or aliases:** Never use "latest" in production
2. **Test before promotion:** Validate challenger models before setting as champion
3. **Monitor performance:** Track prediction metrics over time
4. **Automate retraining:** Schedule periodic model refresh jobs

---

## Troubleshooting

### Common Issues

#### "Model signature not found"

**Cause:** MLflow R client doesn't natively support signatures for UC.

**Solution:** Apply the `mlflow_log_model` patch from the [MLflow Integration](#patching-mlflow_log_model-for-signature-support) section.

---

#### "Column not found" during inference

**Cause:** Input column names don't match the model signature.

**Solution:** Rename columns to match signature exactly:

```r
input_df <- input_df %>%
  rename(
    sepal_length = sepal_length_cm,
    sepal_width = sepal_width_cm
  )
```

---

#### "Cannot register model to Unity Catalog from R"

**Cause:** R MLflow client doesn't support UC registration directly.

**Solution:** Use Python for registration:

```python
%python
import mlflow
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model("runs:/run_id/artifact", "catalog.schema.model")
```

---

#### "Package 'carrier' not found"

**Cause:** Package not pre-installed on Databricks cluster.

**Solution:** Install at the start of your notebook:

```r
install.packages("carrier")
```

---

#### Model predictions fail silently

**Cause:** The crate may not have captured all dependencies.

**Solution:** Explicitly include required objects in the crate:

```r
r_func <- carrier::crate(
  function(newdata) {
    stats::predict(model, newdata = newdata, type = "class")
  },
  model = model,
  additional_param = some_value  # Include any other required objects
)
```

---

## Resources

- [Databricks R Compendium - Log R Models to UC](https://zacdav-db.github.io/dbrx-r-compendium/chapters/mlflow/log-to-uc.html)
- [MLflow R API Documentation](https://mlflow.org/docs/latest/R-api.html)
- [sparklyr Documentation](https://spark.rstudio.com/)
- [Unity Catalog Documentation](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)
- [carrier Package](https://cran.r-project.org/web/packages/carrier/index.html)

