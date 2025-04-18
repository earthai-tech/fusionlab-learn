.. fusionlab documentation master file, created by
   sphinx-quickstart on Thu Apr 17 13:39:49 2025.

.. meta::
   :description: FusionLab: A modular library for Temporal Fusion
                 Transformer (TFT) variants. Extend, experiment,
                 and fuse time-series predictions with
                 state-of-the-art architectures.
   :keywords: time series, forecasting, temporal fusion transformer,
              tft, xtft, machine learning, deep learning, python,
              tensorflow

##################################################
FusionLab: Igniting Next-Gen Fusion Models
##################################################

.. raw:: html

   <p align="center" style="margin-bottom: 1.5em;">
     <a href="https://pypi.org/project/fusionlab/" target="_blank" rel="noopener noreferrer">
       <img src="https://img.shields.io/pypi/v/fusionlab?color=121EAF&label=PyPI" alt="PyPI Version">
     </a>
     <a href="https://fusionlab.readthedocs.io/en/latest/?badge=latest" target="_blank" rel="noopener noreferrer">
       <img src="https://readthedocs.org/projects/fusionlab/badge/?version=latest" alt="Documentation Status"/>
     </a>
     <a href="https://github.com/earthai-tech/fusionlab/blob/main/LICENSE" target="_blank" rel="noopener noreferrer">
       <img src="https://img.shields.io/github/license/earthai-tech/fusionlab?color=121EAF" alt="GitHub License">
     </a>
     <a href="https://www.python.org/" target="_blank" rel="noopener noreferrer">
       <img src="https://img.shields.io/badge/Python-3.8%2B-121EAF" alt="Python Version">
     </a>
     <a href="https://github.com/earthai-tech/fusionlab/actions" target="_blank" rel="noopener noreferrer">
        <img src="https://img.shields.io/github/actions/workflow/status/earthai-tech/fusionlab/python-package-conda.yml?branch=main" alt="Build Status">
     </a>
   </p>

**A Modular Library for Temporal Fusion Transformer (TFT) Variants & Beyond**

*Extend, experiment, and fuse time-series predictions with
state-of-the-art architectures.*

.. raw:: html

    <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

**FusionLab** provides a flexible and extensible framework built on
**TensorFlow/Keras** for advanced time-series forecasting. It centers
on the **Temporal Fusion Transformer (TFT)** and its extensions like
the **Extreme Temporal Fusion Transformer (XTFT)**, offering modular
components and powerful utilities for researchers and practitioners.

Whether you need interpretable multi-horizon forecasts, robust
uncertainty quantification, or a platform to experiment with novel
temporal architectures, FusionLab aims to provide the necessary tools.

.. # --- Sidebar Navigation Structure (Hidden from main page content) ---
.. # This builds the navigation panel on the left.
.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Documentation

   self
   motivation
   installation
   quickstart
   user_guide/index
   user_guide/examples/index
   api
   contributing
   code_of_conduct
   license
   citing
   glossary
   release_notes

.. # --- Main Content Navigation Grid (using sphinx-design) ---
.. # This provides attractive clickable cards on the main page.

.. container:: sd-text-center sd-mt-4 sd-mb-4

   .. row::
      :gutter: 3 

      .. col::
         :col-md-6: :sd-col-lg-4: 

         .. card::
            :shadow: md
            :class-card: mb-3 

            **🚀 Getting Started**
            
            New to FusionLab? Install the library and run your first
            forecast in minutes.

            .. button-ref:: installation
               :color: primary 
               :outline:
               :expand:

               Installation Guide

            .. button-ref:: quickstart
               :color: primary
               :outline:
               :expand:

               Quickstart Example

      .. col::
         :col-md-6: :col-lg-4:

         .. card::
            :shadow: md
            :class-card: mb-3

            **📘 User Guide**
            
            Dive deeper into core concepts, model architectures,
            components, utilities, and advanced features like anomaly
            detection and tuning.

            .. button-ref:: /user_guide/index
               :ref-type: doc 
               :color: primary
               :outline:
               :expand:

               Explore the User Guide

      .. col::
         :col-md-6: :sd-col-lg-4:

         .. card::
            :shadow: md
            :class-card: mb-3

            **💡 Examples Gallery**
           
            See practical code examples demonstrating various use cases,
            from basic forecasting to complex workflows.

            .. sd-button-ref:: /user_guide/examples/index
               :ref-type: doc
               :color: primary
               :outline:
               :expand:

               View Examples

      .. col::
         :col-md-6: :sd-col-lg-4:

         .. card::
            :shadow: md
            :class-card: mb-3

            **</> API Reference**
         
            Detailed specifications for all public modules, classes,
            functions, and methods. Essential for development.

            .. button-ref:: api
               :color: primary
               :outline:
               :expand:

               Browse the API

      .. col::
         :col-md-6: :col-lg-4:

         .. card::
            :shadow: md
            :class-card: sd-mb-3

            **🤝 Development & Community**
            
            Find out how to contribute, report issues, and understand
            project governance and release history.

            .. button-ref:: contributing
               :color: secondary 
               :outline:
               :expand:

               Contribution Guide

            .. button-ref:: code_of_conduct
               :color: secondary
               :outline:
               :expand:

               Code of Conduct

            .. button-ref:: release_notes
               :color: secondary
               :outline:
               :expand:
         
               Release Notes

      .. col::
         :col-md-6: :col-lg-4:

         .. card::
            :shadow: md
            :class-card: sd-mb-3

            **📜 Reference**
            
            Important reference information including the project
            license, how to cite the software, and a glossary of terms.

            .. button-ref:: license
               :color: secondary
               :outline:
               :expand:

               License (BSD-3-Clause)

            .. button-ref:: citing
               :color: secondary
               :outline:
               :expand:

               How to Cite

            .. button-ref:: glossary
               :color: secondary
               :outline:
               :expand:

               Glossary


.. raw:: html

    <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">


.. card::
    :class-card: sd-outline-info sd-rounded-lg sd-p-4 sd-mb-4

    **✨ Key Features**
    
    * 🧩 **Modular Design:**
      Build custom forecasting models using interchangeable components
      like specialized attention layers, GRNs, VSNs, multi-scale LSTMs,
      and more. Facilitates research and tailored solutions.
    * 🚀 **Advanced Architectures:**
      Includes robust implementations of standard TFT, NTFT (variant),
      the high-capacity **XTFT** for complex scenarios, and experimental
      SuperXTFT. Ready-to-use state-of-the-art models.
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

.. raw:: html

    <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">