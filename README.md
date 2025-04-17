<p align="center">
  <img src="docs/source/_static/fusionlab.svg" alt="FusionLab Logo" width="200">
</p>

-----------------------------------------------------

# FusionLab 🔥🧪: Igniting Next-Gen Fusion Models

### _A Modular Library for Temporal Fusion Transformer (TFT) Variants & Beyond_

*Extend, experiment, and fuse time-series predictions with state-of-the-art architectures.*

[![PyPI Version](https://img.shields.io/pypi/v/fusionlab?color=blue)](https://pypi.org/project/fusionlab/)
[![Documentation Status](https://readthedocs.org/projects/fusionlab/badge/?version=latest)](https://fusionlab.readthedocs.io/en/latest/?badge=latest) ![GitHub License](https://img.shields.io/github/license/earthai-tech/fusionlab)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
![GitHub License](https://img.shields.io/github/license/earthai-tech/fusionlab)
[![Build Status](https://img.shields.io/github/actions/workflow/status/earthai-tech/fusionlab/main.yml?branch=main)](https://github.com/earthai-tech/fusionlab/actions) ---

**FusionLab** provides a flexible and extensible framework in Python
for working with advanced time-series forecasting models. It focuses
on the **Temporal Fusion Transformer (TFT)** architecture and its
powerful extensions like the **Extreme Temporal Fusion Transformer (XTFT)**,
offering reusable components and pre-configured models.

Whether you're a researcher exploring novel architectures or a
practitioner building robust forecasting systems, FusionLab provides
tools built on top of **TensorFlow/Keras** to accelerate your work.

---

## ✨ Key Features

* 🧩 **Modular Components:** Build custom models using reusable blocks:
    * Gated Residual Networks (GRNs)
    * Variable Selection Networks (VSNs)
    * Specialized Attention Layers (Temporal, Cross, Hierarchical, Memory-Augmented)
    * Multi-Scale LSTMs & Multi-Resolution Fusion
    * Learned Normalization, Positional Encoding
    * And more... (See :doc:`Components Guide <user_guide/components>`)
* 🚀 **Advanced Architectures Implemented:**
    * :class:`~fusionlab.nn.TemporalFusionTransformer`: A flexible implementation of the standard TFT.
    * :class:`~fusionlab.nn.NTemporalFusionTransformer`: A variant requiring static/dynamic inputs (point forecasts only currently).
    * :class:`~fusionlab.nn.XTFT`: High-capacity *Extreme Temporal Fusion X* with advanced attention, multi-scale processing, and anomaly detection features.
    * :class:`~fusionlab.nn.SuperXTFT`: An experimental enhancement of XTFT with input VSNs (currently deprecated).
    * *(Others like `TFT` from `_adj_tft` if applicable)*
* 🔬 **Integrated Anomaly Detection:** XTFT includes strategies for
    feature-based or prediction-based anomaly score calculation and
    integration into the loss.
* 🛠️ **Practical Utilities:** Includes helpers for:
    * Time series data preprocessing and validation (`ts_utils`).
    * Sequence generation for training (`create_sequences`, `reshape_xtft_data`).
    * Preparing inputs for future predictions (`prepare_spatial_future_data`).
    * Generating and visualizing forecasts (`generate_forecast`, `visualize_forecasts`).
    * Hyperparameter tuning using Keras Tuner (`forecast_tuner`).
* ⚙️ **TensorFlow Backend:** Currently built on TensorFlow/Keras, leveraging its
    ecosystem. *(Future compatibility with other backends like PyTorch/JAX
    is a design goal but not yet implemented).*


---

## 🚀 Getting Started

### Installation

1.  **Prerequisites:**
    * Python 3.8+
    * TensorFlow 2.x (See [TensorFlow Installation Guide](https://www.tensorflow.org/install))

2.  **Install from PyPI (Recommended):**
    ```bash
    pip install fusionlab
    ```
    *(TensorFlow might need separate installation depending on your system)*

3.  **Install from Source (for Development):**
    ```bash
    git clone [https://github.com/earthai-tech/fusionlab.git](https://github.com/earthai-tech/fusionlab.git)
    cd fusionlab
    pip install -e .
    # Optional: Install dev dependencies
    # pip install -e .[dev]
    ```

### Quick Example

```python
import numpy as np
import tensorflow as tf
from fusionlab.nn import TemporalFusionTransformer # Or XTFT etc.

# --- 1. Prepare Dummy Data ---
# (Replace with your actual preprocessed & sequenced data)
B, T, D_dyn = 16, 10, 3  # Batch, TimeSteps, DynamicFeatures
D_stat = 2              # StaticFeatures
D_fut = 1               # FutureFeatures
H = 5                   # Forecast Horizon

# Model expects list: [Static, Dynamic, Future] (if available)
dummy_static = np.random.rand(B, D_stat).astype(np.float32)
dummy_dynamic = np.random.rand(B, T, D_dyn).astype(np.float32)
dummy_future = np.random.rand(B, T + H, D_fut).astype(np.float32) # Needs horizon length
dummy_target = np.random.rand(B, H, 1).astype(np.float32) # Point forecast

# Prepare inputs list (adjust if not using all input types)
# Note: Real data prep involves tools like reshape_xtft_data
model_inputs = [dummy_static, dummy_dynamic, dummy_future]

# --- 2. Instantiate Model ---
# (Using simple TFT for this example)
model = TemporalFusionTransformer(
    static_input_dim=D_stat,
    dynamic_input_dim=D_dyn,
    future_input_dim=D_fut,
    forecast_horizon=H,
    hidden_units=16, # Smaller units for quick example
    num_heads=2
    # quantiles=None # for point forecast (default)
)

# --- 3. Compile & Train ---
model.compile(optimizer='adam', loss='mse')
print("Training simple model...")
model.fit(model_inputs, dummy_target, epochs=2, batch_size=4, verbose=0)
print("Training finished.")

# --- 4. Predict ---
# (Use appropriately prepared inputs for prediction)
# For simplicity, predict on same inputs (not recommended practice)
print("Making predictions...")
predictions = model.predict(model_inputs)
print("Prediction shape:", predictions.shape)
# Expected: (16, 5, 1) -> (Batch, Horizon, NumOutputs)
````

*(See the :doc:`Quickstart Guide <quickstart>` for a more detailed walkthrough.)*

-----

## 📚 Documentation

For detailed usage, tutorials, API reference, and explanations of the
underlying concepts, please see the full documentation:

**[Read the Documentation](https://www.google.com/search?q=https://fusionlab.readthedocs.io/en/latest/)**

-----

## 🤝 Contributing

We welcome contributions\! Whether it's adding new features, fixing bugs,
or improving documentation, your help is appreciated. Please see our
[Contribution Guidelines](https://www.google.com/search?q=CONTRIBUTING.md) for more details on how to get
started.

-----

## 📄 License

This project is licensed under the ** BSD-3-Clause**. See the
[LICENSE](https://www.google.com/search?q=LICENSE) file for details.


## Note 
**🌟 Stay in the Fusion Loop!**  
*Click the "Watch" button above to get notified about the next release — exciting enhancements, new architectures, and experimental features are brewing! 🚀*

