.. fusionlab documentation master file, created by
   sphinx-quickstart on Thu Apr 17 13:39:49 2025.
   You can adapt this file completely to your liking, but it should
   at least contain the root `toctree` directive.

.. meta::
   :description: FusionLab: A modular library for Temporal Fusion
                 Transformer (TFT) variants. Extend, experiment,
                 and fuse time-series predictions with
                 state-of-the-art architectures.
   :keywords: time series, forecasting, temporal fusion transformer,
              tft, xtft, machine learning, deep learning, python,
              tensorflow, pytorch, jax

.. image:: _static/logo.png
   :alt: FusionLab Logo
   :align: center
   :width: 150px

.. raw:: html

   <p align="center">
       <a href="https://pypi.org/project/fusionlab/">
           <img src="https://img.shields.io/pypi/v/fusionlab?color=blue" alt="PyPI Version">
       </a>
       <a href="https://github.com/earthai-tech/fusionlab/blob/main/LICENSE">
           <img src="https://img.shields.io/github/license/earthai-tech/fusionlab" alt="GitHub License">
       </a>
       <a href="https://www.python.org/">
           <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python Version">
       </a>
   </p>

==================================================
FusionLab 🔥🧪: Igniting Next-Gen Fusion Models
==================================================

**A Modular Library for Temporal Fusion Transformer (TFT) Variants & Beyond**

*Extend, experiment, and fuse time-series predictions with
state-of-the-art architectures.*

---

**FusionLab** provides a flexible and extensible framework in
Python for working with advanced time-series forecasting models,
with a special focus on the Temporal Fusion Transformer (TFT) and
its powerful extensions like the Extreme Temporal Fusion
Transformer (XTFT). Whether you're a researcher exploring novel
architectures or a practitioner building robust forecasting
systems, FusionLab offers the tools you need.

---

**Key Features:**
---------------

* 🧩 **Modular Design:** Build custom TFT variants using reusable,
  well-defined components (attention layers, GRNs, VSNs,
  multi-scale LSTMs, etc.).

* 🚀 **Advanced Architectures:** Includes implementations of standard
  TFT and the high-capacity **XTFT** (Extreme Temporal Fusion X)
  for complex time-series challenges, featuring memory-augmented
  attention, multi-resolution fusion, and more.

* 💡 **Extensible:** Easily add new model variants, custom layers,
  or loss functions to experiment with cutting-edge ideas.

* ⚙️ **Framework Compatibility:** Designed with compatibility in mind
  for major deep learning frameworks like TensorFlow (current
  focus), PyTorch, and JAX. *(Specify primary framework support)*

* 🛠️ **Utilities:** Comes with helpful utilities for time-series
  preprocessing, data loading, and evaluation.

* 🔬 **Anomaly Detection:** Integrated anomaly detection capabilities
  within XTFT for identifying and leveraging irregularities in
  data.

---

**Get Started:**
------------------

New to FusionLab? Start here:

* :doc:`installation`: How to install the library.
* :doc:`quickstart`: A quick example to get you forecasting in
  minutes.

---

**Dive Deeper:**
------------------

Explore the core concepts and capabilities:

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   user_guide/introduction
   user_guide/models
   user_guide/components
   user_guide/anomaly_detection
   user_guide/examples

---

**API Reference:**
---------------------

The complete reference for all modules, classes, and functions:

.. toctree::
   :maxdepth: 2
   :caption: API Documentation:

   api

---

**Community & Development:**
------------------------------

* `GitHub Repository <https://github.com/earthai-tech/fusionlab>`_
  Check out the source code, report issues, and contribute.
* :doc:`contributing`: Guidelines for contributing to FusionLab.
* :doc:`license`: Project license information (BSD-3-CLAUSE).