.. _user_guide_forecast_tuner:

=======================
Hyperparameter Tuning
=======================

Finding the optimal set of hyperparameters for deep learning models
like :class:`~fusionlab.nn.transformers.TemporalFusionTransformer`,
:class:`~fusionlab.nn.transformers.TFT` (stricter version),
:class:`~fusionlab.nn.XTFT`, and
:class:`~fusionlab.nn.SuperXTFT` is crucial for achieving the best
possible forecasting performance. Hyperparameters control aspects of
the model architecture (e.g., number of hidden units, attention
heads) and the training process (e.g., learning rate, batch size).

``fusionlab`` provides utility functions within the
:mod:`~fusionlab.nn.forecast_tuner` module that leverage the
powerful **Keras Tuner** library (`keras-tuner`) to automate this
search process.


.. toctree::
   :maxdepth: 1
   :caption: Guide

   forecast_tuner
   forecast_tuner_class_based

.. toctree::
   :maxdepth: 1
   :caption: Tuning Examples:
   
   tuning_examples
   tuning_xtft

.. note::
   The tuning examples use small search spaces and few trials for
   demonstration purposes. For real-world applications, you'll likely
   want to explore a wider range of hyperparameters and run the
   tuner for more trials and epochs to find the best configurations
   for your specific dataset and task.

