.. _forecast_tuner_guide:

===========================
Hyperparameter Tuning
===========================

Finding the optimal set of hyperparameters is one of the most critical
steps in achieving peak performance with advanced forecasting models.
The ``fusionlab-learn`` library provides a powerful and flexible tuning
framework built on top of the industry-standard `Keras Tuner
<https://keras.io/keras_tuner/>`_ library.

Our tuning utilities are designed to automate the search for the best
model architecture and training configurations, saving you significant
time and effort. This section provides detailed guides and practical
examples for each of the available tuners.

The guides are organized by the model families they are designed to
optimize.

.. toctree::
   :maxdepth: 2
   :caption: PINN Model Tuning

   hydro_tuner_guide
   hydro_tuner_examples

.. toctree::
   :maxdepth: 2
   :caption: Hybrid & Transformer Model Tuning

   halnet_tuner_guide
   tft_and_xtft_forecast_tuner_guide
   tft_and_xtft_tuning_examples
   xtft_tuning_examples

.. toctree::
   :maxdepth: 2
   :caption: Other Guides & Examples

   tft_and_xtft_class_based_tuner_guide
   pihalnet_tuning_examples

.. toctree::
   :maxdepth: 2
   :caption: Legacy Tuners

   pihal_tuner_legacy_guide
   

.. note::
   The tuning examples use small search spaces and few trials for
   demonstration purposes. For real-world applications, you'll likely
   want to explore a wider range of hyperparameters and run the
   tuner for more trials and epochs to find the best configurations
   for your specific dataset and task.