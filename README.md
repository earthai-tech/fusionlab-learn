<p align="center">
  <img src="docs/source/_static/fusionlab.svg" alt="FusionLab Logo" width="200">
</p>

-----------------------------------------------------

<h1 align="center">fusionlab-learn</h1>

<p align="center"><em>🔥🧪 A Research-Oriented Library for Advanced Time Series Forecasting with Hybrid, Transformer, and Physics-Informed Models</em></p>

<p align="center">
  <a href="https://pypi.org/project/fusionlab-learn/"><img src="https://img.shields.io/pypi/v/fusionlab-learn" alt="PyPI Version"></a>
  <a href="https://fusion-lab.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/fusion-lab/badge/?version=latest" alt="Documentation Status"></a>
  <a href="https://github.com/earthai-tech/fusionlab-learn/actions"><img src="https://img.shields.io/github/actions/workflow/status/earthai-tech/fusionlab-learn/.github%2Fworkflows%2Fpython-package-conda.yml" alt="Build Status"></a>
  <a href="https://www.python.org/downloads/release/python-390/"><img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python Version"></a>
  <a href="https://github.com/earthai-tech/fusionlab-learn/blob/main/LICENSE"><img src="https://img.shields.io/github/license/earthai-tech/fusionlab-learn?style=flat&color=cyan" alt="License"></a>
</p>

**fusionlab-learn** is a flexible and extensible Python package for building and experimenting with state-of-the-art time series models. It provides robust, research-grade implementations of advanced architectures, from data-driven forecasters to novel Physics-Informed Neural Networks (PINNs).

Whether you're a researcher exploring new architectures or a practitioner building production-grade forecasting systems, `fusionlab-learn` provides tools built on **TensorFlow/Keras** to accelerate your work.

---

## ✨ Key Features

### 🏛️ A Spectrum of Advanced Architectures
The library provides implementations across three major families of forecasting models.

* **[Hybrid Models](https://fusion-lab.readthedocs.io/en/latest/user_guide/models/hybrid/index.html):** Architectures like `HALNet` and `XTFT` that fuse the sequential processing power of LSTMs with the long-range context modeling of attention mechanisms.
* **[Pure Transformers](https://fusion-lab.readthedocs.io/en/latest/user_guide/models/transformers/index.html):** Implementations of the standard "Attention Is All You Need" encoder-decoder architecture, adapted for time series forecasting.
* **[Physics-Informed Models (PINNs)](https://fusion-lab.readthedocs.io/en/latest/user_guide/models/pinn/index.html):** State-of-the-art hybrid models like `TransFlowSubsNet` that integrate physical laws (PDEs) directly into the training process to produce physically consistent and robust forecasts.

### 🧩 Modular & Reusable Components
Build custom models with a rich set of well-tested neural network blocks, including:
* [Gated Residual Networks (GRNs) & Variable Selection Networks (VSNs)](https://fusion-lab.readthedocs.io/en/latest/user_guide/components.html)
* Specialized [Attention Layers](https://fusion-lab.readthedocs.io/en/latest/user_guide/user_guide/components.html#attention-mechanisms): `CrossAttention`, `HierarchicalAttention`, and `MemoryAugmentedAttention`
* [Multi-Scale LSTMs](https://fusion-lab.readthedocs.io/en/latest/user_guide/components.html#multiscalelstm) for capturing temporal patterns at various resolutions.

### ⚛️ PINN Capabilities
-   Solve coupled-physics problems with models like **[TransFlowSubsNet](https://fusion-lab.readthedocs.io/en/latest/user_guide/models/pinn/transflow_subnet.html)**.
-   Perform **inverse modeling** by configuring physical coefficients (`K`, `Ss`, `C`) as learnable parameters.
-   Utilize specialized **[PINN data utilities](https://fusion-lab.readthedocs.io/en/latest/user_guide/utils/pinn_utils.html)** for the unique sequence and coordinate preparation required by these models.

### 🛠️ Unified Hyperparameter Tuning
-   Leverage the **[HydroTuner](https://fusion-lab.readthedocs.io/en/latest/user_guide/forecast_tuner/hydro_tuner_guide.html)** to automatically find optimal hyperparameters for all hydrogeological PINN models.
-   Use dedicated tuners for data-driven models like `HALNet` and `XTFT`.
-   The tuner's `.create()` factory method automatically infers data dimensions, making setup fast and easy.

---

## 🚀 Getting Started

### Installation

1.  **Prerequisites:**
    * Python 3.9+
    * [TensorFlow >=2.15](https://www.tensorflow.org/install)

2.  **Install from PyPI (Recommended):**
    ```bash
    pip install fusionlab-learn
    ```

3.  **Install from Source (for Development):**
    ```bash
    git clone https://github.com/earthai-tech/fusionlab-learn.git
    cd fusionlab-learn
    pip install -e .
    ```

### Quick Example

```python
import numpy as np
import tensorflow as tf
from fusionlab.nn.models import HALNet # Or any other model

# --- 1. Prepare Dummy Data ---
# (Replace with your actual preprocessed & sequenced data)
B, T, D_dyn = 16, 10, 3  # Batch, TimeSteps, DynamicFeatures
D_stat = 2               # StaticFeatures
D_fut = 1                # FutureFeatures
H = 5                    # Forecast Horizon

# Model expects list: [Static, Dynamic, Future]
dummy_static = np.random.rand(B, D_stat).astype(np.float32)
dummy_dynamic = np.random.rand(B, T, D_dyn).astype(np.float32)
# For 'tft_like' mode, future input spans past + horizon
dummy_future = np.random.rand(B, T + H, D_fut).astype(np.float32)
dummy_target = np.random.rand(B, H, 1).astype(np.float32)

model_inputs = [dummy_static, dummy_dynamic, dummy_future]

# --- 2. Instantiate Model ---
model = HALNet(
    static_input_dim=D_stat,
    dynamic_input_dim=D_dyn,
    future_input_dim=D_fut,
    forecast_horizon=H,
    max_window_size=T,
    output_dim=1,
    hidden_units=16, # Smaller units for quick example
    num_heads=2
)

# --- 3. Compile & Train ---
model.compile(optimizer='adam', loss='mse')
print("Training simple model...")
model.fit(model_inputs, dummy_target, epochs=2, batch_size=4, verbose=0)
print("Training finished.")

# --- 4. Predict ---
print("Making predictions...")
predictions = model.predict(model_inputs)
print("Prediction shape:", predictions.shape)
# Expected: (16, 5, 1) -> (Batch, Horizon, NumOutputs)

```

*(See the* [*Quickstart Guide*](https://fusion-lab.readthedocs.io/en/latest/quickstart.html) *for a more detailed walkthrough.)*

-----

## 📚 Documentation

For detailed usage, tutorials, API reference, and explanations of the
underlying concepts, please see the full documentation:

**[Read the Documentation](https://fusion-lab.readthedocs.io/)**

-----


## 📄 License

This project is licensed under the **BSD-3-Clause**. See the
[LICENSE](https://github.com/earthai-tech/fusionlab-learn/blob/main/LICENSE) file for details.

----

## 🤝 Contributing

We welcome contributions\! Whether it's adding new features, fixing bugs,
or improving documentation, your help is appreciated. Please see our
[Contribution Guidelines](https://fusion-lab.readthedocs.io/en/latest/contributing.html) for more details on how to get
started.

-----

## 📞 Contact & Support

  * **Bug Reports & Feature Requests:** The best place to report issues,
    ask questions about usage, or request new features is the
    [**GitHub Issues**](https://github.com/earthai-tech/fusionlab-learn/issues) page for the project.

  * **Developer Contact:** For direct inquiries related to the project's
    origins or specific collaborations, you can reach the author:

      * **Name:** Laurent Kouadio
      * 📧 **Email:** [etanoyau@gmail.com](mailto:etanoyau@gmail.com)
      * 💼 **LinkedIn:** [linkedin.com/in/laurent-kouadio-483b2baa](https://linkedin.com/in/laurent-kouadio-483b2baa)
      * 🆔 **ORCID:** [0000-0001-7259-7254](https://orcid.org/0000-0001-7259-7254)
