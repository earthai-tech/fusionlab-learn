. _gallery_hyperparameter_tuning_index:

================================
Hyperparameter Tuning Examples
================================

This section of the gallery provides practical examples of using the
hyperparameter tuning utilities in ``fusionlab-learn``, primarily
leveraging `Keras Tuner`.

Finding the right set of hyperparameters is crucial for achieving
optimal performance with complex deep learning models like TFT and
XTFT. These examples will guide you through setting up and running
the tuning process.

.. toctree::
   :maxdepth: 1
   :caption: Tuning Examples:

   hyperparameter_tuning_example
   hyperparameter_tuning_for_xtft
   hyperparameter_tuning_for_superxtft

.. note::
   The tuning examples use small search spaces and few trials for
   demonstration purposes. For real-world applications, you'll likely
   want to explore a wider range of hyperparameters and run the
   tuner for more trials and epochs to find the best configurations
   for your specific dataset and task.

