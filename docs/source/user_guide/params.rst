.. _params_guide:

================================
Physical Parameter Descriptors
================================

The ``fusionlab.params`` module provides a suite of simple,
self-documenting classes designed to give you explicit control over how
physical coefficients are handled within the library's Physics-Informed
Neural Networks (PINNs).

Instead of using ambiguous strings like ``'learnable'`` or bare floats,
these classes make the model's configuration clear and robust. They
allow you to specify whether a physical parameter should be treated as
a fixed constant or as a trainable variable to be discovered by the model
during training (inverse modeling).

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

General Learnable Parameters (`BaseLearnable`)
------------------------------------------------
This is the modern and recommended approach for defining any physical
parameter in models like ``TransFlowSubsNet``. The ``BaseLearnable``
class is an abstract base that provides a consistent framework for
creating learnable or fixed physical constants.

**Key Features:**

* **Explicit Control:** Clearly defines whether a parameter is
  trainable.
* **Positivity Constraints:** Can enforce positivity for parameters
  like hydraulic conductivity (:math:`K`) by learning their logarithm,
  :math:`\log(K)`, and then taking the exponent during use.
* **Keras Serialization:** Fully compatible with Keras model saving
  and loading.

`LearnableK`
**************
:API Reference: :class:`~fusionlab.params.LearnableK`

A specific implementation for defining the **Hydraulic Conductivity**
(:math:`K`). By default, it uses a log-transform to ensure the learned
value of :math:`K` is always positive.

**Usage Example**

.. code-block:: python
   :linenos:

   from fusionlab.params import LearnableK

   # Define K as a trainable variable, starting from an initial
   # guess of 0.001. The model will learn its optimal value.
   k_parameter = LearnableK(initial_value=0.001)

   # In the model, you would pass this object:
   # model = TransFlowSubsNet(..., K=k_parameter, ...)

`LearnableSs`
******************
:API Reference: :class:`~fusionlab.params.LearnableSs`

An implementation for defining the **Specific Storage** (:math:`S_s`).
Like `LearnableK`, it defaults to using a log-transform to enforce
positivity.

**Usage Example**

.. code-block:: python
   :linenos:

   from fusionlab.params import LearnableSs

   # Define Ss as a trainable variable with an initial value.
   ss_parameter = LearnableSs(initial_value=1e-5)

`LearnableQ`
******************
:API Reference: :class:`~fusionlab.params.LearnableQ`

An implementation for defining the **Source/Sink Term** (:math:`Q`).
Unlike :math:`K` and :math:`S_s`, the source/sink term can be positive
(injection) or negative (pumping). Therefore, this class does **not**
use a log-transform by default and learns the value directly.

**Usage Example**

.. code-block:: python
   :linenos:

   from fusionlab.params import LearnableQ

   # Define Q as a trainable variable, starting from zero.
   q_parameter = LearnableQ(initial_value=0.0)


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Consolidation Coefficient Descriptors (`_BaseC`)
------------------------------------------------------
These are the original, more specialized descriptor classes designed
specifically for the consolidation coefficient (:math:`C`) used in
``PIHALNet``. While the general `BaseLearnable` is now preferred, these
remain useful and provide clear, self-documenting intent.

`LearnableC`
******************
:API Reference: :class:`~fusionlab.params.LearnableC`

This class signals that the consolidation coefficient :math:`C` should be
a trainable variable. It learns :math:`\log(C)` to ensure :math:`C > 0`.

**Usage Example**

.. code-block:: python
   :linenos:

   from fusionlab.params import LearnableC

   # Configure the model to discover the value of C during training.
   learnable_c = LearnableC(initial_value=0.01)
   # model = PIHALNet(..., pinn_coefficient_C=learnable_c)

`FixedC`
******************
:API Reference: :class:`~fusionlab.params.FixedC`

This class signals that the coefficient :math:`C` should be treated as
a non-trainable, fixed constant.

**Usage Example**

.. code-block:: python
   :linenos:

   from fusionlab.params import FixedC

   # Use a known, fixed value for C.
   fixed_c = FixedC(value=0.05)
   # model = PIHALNet(..., pinn_coefficient_C=fixed_c)

`DisabledC`
******************
:API Reference: :class:`~fusionlab.params.DisabledC`

This class is a simple flag used to signal that the physics-informed
loss related to the coefficient :math:`C` should be completely
disabled. This is useful for running the model in a purely data-driven
mode for ablation studies.

**Usage Example**

.. code-block:: python
   :linenos:

   from fusionlab.params import DisabledC

   # Run the model without the consolidation physics constraint.
   # Note: The `lambda_physics` weight in .compile() should also be 0.
   no_physics_c = DisabledC()
   # model = PIHALNet(..., pinn_coefficient_C=no_physics_c)

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Parameter Resolution Utility (`resolve_physical_param`)
-------------------------------------------------------
:API Reference: :func:`~fusionlab.params.resolve_physical_param`

This is a powerful internal utility that normalizes the various ways a
user might specify a physical parameter (e.g., a simple float, the
string ``'learnable'``, or a `LearnableK` instance) into a consistent
internal representation that the model can use. While you typically won't
call this function directly, it's what allows the model constructors to
be so flexible.
