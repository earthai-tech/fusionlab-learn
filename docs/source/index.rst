.. fusionlab documentation master file, created by
   sphinx-quickstart on Thu Apr 17 13:39:49 2025.

.. meta::
   :description: FusionLab: Modular library for Temporal Fusion Transformer
                 (TFT) variants. Extend, experiment, and fuse time-series
                 predictions with state-of-the-art architectures.
   :keywords: time series, forecasting, temporal fusion transformer, tft,
              xtft, machine learning, deep learning, python, tensorflow

################################################################
:code:`Fusionlab-learn`: Igniting Next-Gen Fusion Models
################################################################

.. raw:: html

   <p align="center" style="margin-bottom: 1.5em;">
     <a href="https://pypi.org/project/fusionlab-learn/" target="_blank" rel="noopener noreferrer">
       <img src="https://img.shields.io/pypi/v/fusionlab-learn?color=121EAF&label=PyPI" alt="PyPI Version">
     </a>
     <a href="https://fusion-lab.readthedocs.io/en/latest/?badge=latest" target="_blank" rel="noopener noreferrer">
       <img src="https://readthedocs.org/projects/fusion-lab/badge/?version=latest" alt="Documentation Status"/>
     </a>
     <a href="https://github.com/earthai-tech/fusionlab-learn/blob/main/LICENSE" target="_blank" rel="noopener noreferrer">
       <img src="https://img.shields.io/github/license/earthai-tech/fusionlab-learn?color=121EAF" alt="GitHub License">
     </a>
     <a href="https://www.python.org/" target="_blank" rel="noopener noreferrer">
       <img src="https://img.shields.io/badge/Python-3.9%2B-121EAF" alt="Python Version">
     </a>
     <a href="https://github.com/earthai-tech/fusionlab-learn/actions" target="_blank" rel="noopener noreferrer">
        <img src="https://img.shields.io/github/actions/workflow/status/earthai-tech/fusionlab-learn/python-package-conda.yml?branch=main" alt="Build Status">
     </a>
   </p>

.. container:: special-card-wrapper

   .. card:: **A Modular Library for Temporal Fusion Transformer (TFT) Variants & Beyond**
      :margin: 0 0 1 0
      :text-align: center

      *Extend, experiment, and fuse time-series predictions with
      state-of-the-art architectures.*


.. raw:: html

    <hr style="margin-top: 1.5em; margin-bottom: 2em;">

**Fusionlab-learn** provides a flexible and extensible framework built on
**TensorFlow/Keras** for advanced time-series forecasting. It centers
on the **Temporal Fusion Transformer (TFT)** and its extensions like
the **Extreme Temporal Fusion Transformer (XTFT)**, offering modular
components and powerful utilities for researchers and practitioners.

Whether you need interpretable multi-horizon forecasts, robust
uncertainty quantification, or a platform to experiment with novel
temporal architectures, FusionLab aims to provide the necessary tools.

.. container:: text-center

    .. button-ref:: installation
        :color: primary
        :expand:
        :outline:

        Install fusion-lab
        
.. container:: button-container

    .. grid:: 1 2 2 2
       :gutter: 2

       .. grid-item-card:: Quickstart
          :link: quickstart
          :link-type: doc
          :class-item: button-link-primary

       .. grid-item-card:: User Guide
          :link: /user_guide/index
          :link-type: doc
          :class-item: button-link-secondary

       .. grid-item-card:: API Reference
          :link: api
          :link-type: doc
          :class-item: button-link-secondary

       .. grid-item-card:: What's New?
          :link: /release_notes/index
          :link-type: doc
          :class-item: button-link-primary sparkle-link
          

.. topic:: Key Features
   :class: sd-rounded-lg

   * 🧩 **Modular Design:**
     Build custom forecasting models using interchangeable components
     like specialized attention layers, GRNs, VSNs, multi-scale LSTMs,
     and more. Facilitates research and tailored solutions.
   * 🚀 **Advanced Architectures:**
     Includes robust implementations of standard TFT, the high-capacity
     **XTFT** for complex scenarios, and experimental SuperXTFT.
     Ready-to-use state-of-the-art models.
   * 💡 **Extensible:**
     Designed for extension. Easily integrate new model architectures,
     custom layers, or novel loss functions to push the boundaries
     of time series forecasting.
   * ⚙️ **TensorFlow Backend:**
     Currently leverages the power and scalability of the TensorFlow/Keras
     ecosystem for building and training models.
   * 🛠️ **Comprehensive Utilities:**
     Offers a suite of helper tools for common tasks: data preparation,
     sequence generation, time series analysis, result visualization,
     hyperparameter tuning, and CLI applications.
   * 🔬 **Anomaly Detection:**
     Features integrated anomaly detection mechanisms within XTFT,
     allowing models to identify and potentially adapt to irregular data
     patterns during training.



.. rubric:: Explore Further

* **Motivation:** Understand the :doc:`motivation` behind FusionLab.
* **Examples:** See practical applications in the :doc:`Examples Gallery </user_guide/gallery/index>`.
* **Contribute:** Learn how to :doc:`contribute <contributing>` to the project.
* **Cite:** Find out :doc:`how to cite <citing>` FusionLab in your work.
* **Reference:** Consult the :doc:`glossary` or view the :doc:`license` (BSD-3-Clause).

.. raw:: html

    <hr style="margin-top: 1.0em; margin-bottom: 1.0em;">

.. admonition:: Terminology

   For brevity and consistency, the library will be referred to as ``fusionlab``
   throughout the remainder of the documentation.

.. raw:: html

    <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">
    
.. # --- Sidebar Navigation Structure (Hidden from main page content) ---
.. # This builds the navigation panel on the left using Furo theme.
.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Documentation

   installation
   quickstart
   motivation
   user_guide/index
   user_guide/examples/index
   api
   contributing
   code_of_conduct
   citing
   release_notes/index
   glossary
   license
   

