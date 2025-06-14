.. _user_guide_pitgwflow:

=========================================================
Physics-Informed Transient Groundwater Flow (PiTGWFlow)
=========================================================

:API Reference: :class:`~fusionlab.nn.pinn.PiTGWFlow`

The Physics-Informed Transient Groundwater Flow model (``PiTGWFlow``)
is a self-contained neural network designed to solve the 2D
transient groundwater flow equation. It serves as a prime example
of a Physics-Informed Neural Network (PINN) that learns directly
from the governing physical laws, rather than from labeled data.

``PiTGWFlow`` uses a simple Multi-Layer Perceptron (MLP) to
approximate the hydraulic head :math:`h` as a continuous and
differentiable function of time :math:`t` and space :math:`(x, y)`.
The model is trained by minimizing the residual of the PDE itself,
making it a powerful tool for solving forward and inverse problems in
hydrology when data is scarce.

Key Features
------------

* **Self-Contained Physics Solver:** The entire model, including
    the neural network surrogate and the physics-based loss
    calculation, is encapsulated in a single, easy-to-use Keras
    model.

* **PDE-Based Loss Function:** The model's objective is to minimize
    the residual of the groundwater flow equation. This unsupervised
    approach allows it to be trained on a set of unlabeled
    "collocation points" sampled from the domain.

* **Learnable Physical Parameters:** Key physical coefficients like
    hydraulic conductivity (:math:`K`) and specific storage
    (:math:`S_s`) can be defined as fixed constants or as
    trainable variables. This enables the model to solve inverse
    problems, inferring physical properties that best satisfy the
g   overning equation.

* **Seamless Keras Integration:** ``PiTGWFlow`` overrides the
    default ``train_step`` and ``test_step`` methods. This allows
    users to train and evaluate the model using the standard Keras
    ``.fit()`` and ``.evaluate()`` API, without needing to manually
    handle the complex gradient calculations for the PDE.

When to Use PiTGWFlow
---------------------

``PiTGWFlow`` is an excellent choice for hydrogeological problems
where:

* You need to solve the **forward problem**: predicting the
    hydraulic head :math:`h(t, x, y)` over a domain when the
    physical parameters (:math:`K`, :math:`S_s`, :math:`Q`) are known.

* You need to solve the **inverse problem**: inferring unknown
    physical parameters like :math:`K` or :math:`S_s` by training
    the model to satisfy the PDE.

* You lack sufficient labeled data for a traditional supervised
    model but have a well-defined physical domain and governing
    equation.

* You require a **continuous and differentiable** solution that can
    be evaluated at any coordinate :math:`(t, x, y)` within the
    trained domain, enabling the calculation of flow velocities or
    other derived quantities.

Architectural Workflow
~~~~~~~~~~~~~~~~~~~~~~~~
``PiTGWFlow``'s architecture is straightforward, combining a
standard neural network with a custom, physics-driven training loop.

**1. The Neural Network Surrogate**

The core of the model is a simple Multi-Layer Perceptron (MLP) that
acts as a universal function approximator. This network takes the
concatenated coordinates :math:`[t, x, y]` as input and outputs a
single scalar value representing the predicted hydraulic head
:math:`h_{NN}`.

.. math::
    h_{NN} = \text{MLP}(\theta; [t, x, y])

Here, :math:`\theta` represents all the trainable weights and biases
of the MLP.

**2. Physics-Informed Loss Calculation (Inside `train_step`)**

This is where the "physics" is injected. Instead of comparing the
output to a ground-truth label, the training step calculates how
well the network's output satisfies the governing PDE.

* **Automatic Differentiation:** For a batch of collocation
    points, the model uses ``tf.GradientTape`` to compute the
    first and second-order derivatives of the network's output
    :math:`h_{NN}` with respect to its inputs :math:`t, x, y`.

    .. math::
       \frac{\partial h_{NN}}{\partial t}, \quad
       \frac{\partial h_{NN}}{\partial x}, \quad
       \frac{\partial^2 h_{NN}}{\partial x^2}, \quad
       \frac{\partial^2 h_{NN}}{\partial y^2}

* **Residual Calculation:** These computed derivatives are plugged
    back into the governing equation to calculate the PDE residual,
    :math:`R`.

    .. math::
       R = S_s \frac{\partial h_{NN}}{\partial t} - K \left(
       \frac{\partial^2 h_{NN}}{\partial x^2} +
       \frac{\partial^2 h_{NN}}{\partial y^2} \right) - Q

* **Loss Computation:** The final loss, :math:`\mathcal{L}`, is
    the mean of the squared residuals over the batch. The goal of
    training is to drive this loss to zero.

    .. math::
       \mathcal{L}(\theta, K, S_s, Q) = \text{mean}(R^2)

**3. Gradient Application**

The gradient of the loss :math:`\mathcal{L}` is computed with
respect to all trainable variables in the model. This includes the
network's weights :math:`\theta` and any parameters defined as
`Learnable` (e.g., `LearnableK`). The optimizer then updates these
variables to minimize the loss.

Complete Example
----------------

This example demonstrates a complete workflow for solving a forward
problem with ``PiTGWFlow``.

Step 1: Imports and Setup
~~~~~~~~~~~~~~~~~~~~~~~~~
First, we import all necessary libraries.

.. code-block:: python
   :linenos:

   import os
   import numpy as np
   import tensorflow as tf
   import matplotlib.pyplot as plt

   # FusionLab imports
   from fusionlab.nn.pinn import PiTGWFlow
   from fusionlab.params import LearnableK
   from fusionlab.nn.models.utils import plot_history_in

   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress logs

   EXERCISE_OUTPUT_DIR = "./pitgwflow_exercise_outputs"
   os.makedirs(EXERCISE_OUTPUT_DIR, exist_ok=True)


Step 2: Generate Collocation Points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We generate a set of random points within our domain. These points
serve as the "training data" where the PDE will be enforced.

.. code-block:: python
   :linenos:

   # Configuration
   N_POINTS = 2000
   BATCH_SIZE = 64

   # Generate collocation points
   tf.random.set_seed(42)
   coords = {
       "t": tf.random.uniform((N_POINTS, 1), 0, 10), # Time from 0 to 10
       "x": tf.random.uniform((N_POINTS, 1), -1, 1), # x from -1 to 1
       "y": tf.random.uniform((N_POINTS, 1), -1, 1), # y from -1 to 1
   }

   # Dummy targets are required for the Keras API but are ignored
   dummy_y = tf.zeros((N_POINTS, 1))

   # Create a tf.data.Dataset
   dataset = tf.data.Dataset.from_tensor_slices(
       (coords, dummy_y)
   ).shuffle(N_POINTS).batch(BATCH_SIZE)

   print(f"Generated {N_POINTS} collocation points.")
   print(f"Dataset element spec: {dataset.element_spec}")


Step 3: Define, Compile, and Train PiTGWFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We instantiate the model, defining one parameter (:math:`K`) as
learnable and the others as fixed. Then, we compile and train it.

.. code-block:: python
   :linenos:

   # Instantiate PiTGWFlow with a learnable K
   pinn_model = PiTGWFlow(
       hidden_units=[40, 40, 40],
       activation='tanh',
       K=LearnableK(1.0), # The model will infer this value
       Ss=1e-4,           # Fixed value
       Q=0.0              # Fixed value
   )

   # Compile and train
   pinn_model.compile()
   print("\nTraining PiTGWFlow model...")
   history = pinn_model.fit(
       dataset,
       epochs=15,
       verbose=0 # Set to 1 to see epoch progress
   )
   print("Training complete.")

**Example Training Output:**

.. code-block:: text

   Training PiTGWFlow model...
   Training complete.
   Final PDE Loss: 1.2345e-05
   Final Learned K: 0.9876

*(Note: The above output is representative. Actual values will vary.)*


Step 4: Visualize Training History
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use a plotting utility to visualize the decrease in the PDE loss.

.. code-block:: python
   :linenos:

   print("\nPlotting training history...")
   fig, ax = plt.subplots(figsize=(8, 5))
   ax.plot(history.history['pde_loss'], label='PDE Loss')
   ax.set_yscale('log')
   ax.set_title('PiTGWFlow Training History')
   ax.set_xlabel('Epoch')
   ax.set_ylabel('Log PDE Loss')
   ax.legend()
   ax.grid(True, which='both', linestyle='--', linewidth=0.5)
   plt.savefig(os.path.join(EXERCISE_OUTPUT_DIR, "pitgwflow_loss_history.png"))
   plt.show()

**Example Output Plot:**

.. figure:: ../images/pitgwflow_loss_history.png
   :alt: PiTGWFlow Training History Plot
   :align: center
   :width: 70%

   An example plot showing the PDE loss decreasing over epochs. The
   logarithmic scale helps visualize the rapid reduction in error as
   the model learns to satisfy the physics.

Next Steps
----------

.. note::

   Now that you understand the theory and basic usage of ``PiTGWFlow``,
   you can apply these concepts in a practical problem.

   Proceed to the exercise: :ref:`user_guide/exercices/exercice_pitgwflow.rst`