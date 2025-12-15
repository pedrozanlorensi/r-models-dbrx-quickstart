# r-models-dbrx-quickstart
A simple example of how you can save an R model to the Unity Catalog and use it for inferences. 

This repo consists of 2 notebooks:
- **1_train_iris_r_classification_model.ipynb** -> In this notebook, you'll train a classification model (using R) on the top of the [Iris Dataset](https://scikit-learn.org/1.4/auto_examples/datasets/plot_iris_dataset.html) and save it to the Unity Catalog.
- **2_run_inferences_with_iris_r_classification_model.ipynb** -> In this notebook, you'll learn how you can load the registered model to run your inferences. 

For more information, check out:
- [Quickstart Doc](./docs/r_models_dbrx_quickstart.md)
- [Instructions Doc](./docs/r_models_dbrx_instructions.md)
