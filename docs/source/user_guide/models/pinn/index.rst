.. _pinn_models_guide:

===========================================
Physics-Informed Neural Networks (PINNs)
===========================================

This section of the user guide delves into the **Physics-Informed
Neural Networks (PINNs)** available within the ``fusionlab-learn``
library. These models represent a cutting-edge approach that merges
the pattern-recognition power of deep learning with the fundamental
principles of physics, expressed as Partial Differential Equations
(PDEs).

Unlike purely data-driven models, PINNs are trained to satisfy both
observational data and the underlying physical laws of a system.
This makes them exceptionally powerful for scientific and
engineering applications where data may be sparse or noisy, as the
physics provides a strong inductive bias, leading to more robust and
generalizable solutions.

The models in this section are designed for complex spatio-temporal
forecasting tasks, such as those found in geohydrology, where
respecting the physical processes is crucial for accurate and
meaningful predictions.

.. toctree::
   :maxdepth: 2
   :caption: PINN Models:

   pihalnet
   transflow_subnet
   pitgwflow