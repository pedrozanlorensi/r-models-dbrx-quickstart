# R Models on Databricks — Quickstart Guide

Get up and running with R models on Databricks in minutes. This guide walks you through training an R model, logging it with MLflow, registering it to Unity Catalog, and running batch inferences.

---

## Prerequisites

- A Databricks workspace with Unity Catalog enabled
- Access to a catalog and schema (e.g., `my_catalog.my_schema`)
- A cluster running with R support (Databricks Runtime ML recommended)

---

## 1. Train & Register an R Model

### Install Required Packages

```r
install.packages("carrier")
```

### Load Libraries

```r
library(sparklyr)
library(dplyr)
library(mlflow)
library(rpart)
library(carrier)
```

### Connect to Spark & Load Data

```r
spark <- spark_connect(method = "databricks")

# Load data from Unity Catalog
iris_tbl <- tbl(spark, "my_catalog.my_schema.iris_data")
iris_df <- collect(iris_tbl)

# Ensure target is a factor for classification
iris_df$species <- as.factor(iris_df$species)
```

### Train a Decision Tree Model

```r
model <- rpart(
  species ~ sepal_length_cm + sepal_width_cm + petal_length_cm + petal_width_cm,
  data = iris_df,
  method = "class"
)
```

### Define Model Signature for Unity Catalog

```r
signature <- list(
  inputs = list(
    list(type = "double", name = "sepal_length"),
    list(type = "double", name = "sepal_width"),
    list(type = "double", name = "petal_length"),
    list(type = "double", name = "petal_width")
  ),
  outputs = list(
    list(type = "string")
  )
)
```

### Patch `mlflow_log_model` for Signature Support

> **Note:** This patch is required for Unity Catalog compatibility.

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
    warning("Logging model metadata failed. Model artifacts logged successfully.")
  })
  res
}

assignInNamespace("mlflow_log_model", mlflow_log_model, ns = "mlflow")
```

### Log Model to MLflow

```r
mlflow_set_experiment("/Users/your_email@domain.com/my_r_experiment")

run <- mlflow_start_run()

# Wrap model in a carrier crate
r_func <- carrier::crate(
  function(newdata) {
    stats::predict(model, newdata = newdata, type = "class")
  },
  model = model
)

mlflow_log_model(r_func, "my_r_model", signature = signature)
mlflow_end_run()

# Get the run ID for registration
print(run$run_id)
```

### Register Model to Unity Catalog (Python)

```python
%python
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")

run_id = "<YOUR_RUN_ID>"  # From R output above
artifact_path = "my_r_model"
full_model_name = "my_catalog.my_schema.my_r_model_uc"

# Register model
registered_model = mlflow.register_model(
    f"runs:/{run_id}/{artifact_path}", 
    full_model_name
)

# Set alias
client = MlflowClient(registry_uri="databricks-uc")
client.set_registered_model_alias(
    name=full_model_name,
    alias="champion",
    version=registered_model.version
)
```

---

## 2. Run Inferences with the Registered Model

### Load the Model

```r
install.packages("mlflow")
install.packages("carrier")

library(sparklyr)
library(dplyr)
library(mlflow)

# Load model from Unity Catalog
model <- mlflow_load_model("models:/my_catalog.my_schema.my_r_model_uc@champion")
```

### Run Predictions

```r
# Load new data
new_data_tbl <- tbl(spark, "my_catalog.my_schema.iris_features")
new_data <- collect(new_data_tbl)

# Run predictions
predictions <- model(new_data)

# Add predictions to dataframe
new_data$prediction <- predictions
```

### Save Results Back to Unity Catalog

```r
# Copy to Spark
results_tbl <- copy_to(spark, new_data, "results_with_preds", overwrite = TRUE)

# Write to Unity Catalog table
spark_write_table(
  results_tbl,
  name = "my_catalog.my_schema.iris_predictions",
  mode = "overwrite"
)
```

---

## Quick Reference

| Action | Code |
|--------|------|
| Log model | `mlflow_log_model(crated_func, "artifact_path", signature = sig)` |
| Load model | `mlflow_load_model("models:/catalog.schema.model@alias")` |
| Run inference | `predictions <- model(new_data)` |
| Write to UC | `spark_write_table(df, name = "catalog.schema.table", mode = "overwrite")` |

---

## Resources

- [R Models in Unity Catalog Blog Post](https://zacdav-db.github.io/dbrx-r-compendium/chapters/mlflow/log-to-uc.html)
- [MLflow R API Docs](https://mlflow.org/docs/latest/R-api.html)
- [Databricks Unity Catalog](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)

