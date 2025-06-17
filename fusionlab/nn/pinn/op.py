# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Physics-Informed Neural Network (PINN) Operations
and Helpers.
"""
from typing import Dict, List, Optional, Tuple, Union, Callable

from ..._fusionlog import fusionlog, OncePerMessageFilter 
from ...utils.deps_utils import ensure_pkg 
from .. import KERAS_DEPS, KERAS_BACKEND,dependency_message 
from .utils import extract_txy_in 

if KERAS_BACKEND:
    Model = KERAS_DEPS.Model
    Tensor = KERAS_DEPS.Tensor
    GradientTape = KERAS_DEPS.GradientTape
    
    tf_concat = KERAS_DEPS.concat
    tf_square = KERAS_DEPS.square 
    tf_name_scope = KERAS_DEPS.name_scope
    tf_float32 =KERAS_DEPS.float32 
    tf_constant = KERAS_DEPS.constant 
    
else:
    class Model:
        pass
    Tensor = type("Tensor", (), {})

DEP_MSG = dependency_message('nn.transformers') 
logger = fusionlog().get_fusionlab_logger(__name__)
logger.addFilter(OncePerMessageFilter())

def compute_consolidation_residual(
    s_pred: Tensor,
    h_pred: Tensor,
    time_steps: Tensor,
    C: Union[float, Tensor], 
    eps: float  = 1e-5
) -> Tensor:
    """
    Computes the residual of a simplified consolidation equation.

    This function enforces a simplified form of Terzaghi's 1D
    consolidation theory on the output sequences of a forecasting
    model. It relates the rate of subsidence :math:`s` to the rate
    of change of hydraulic head :math:`h` (GWL).

    Parameters
    ----------
    s_pred : tf.Tensor
        The predicted subsidence sequence from the model. Expected
        shape is ``(batch_size, time_horizon, 1)``.
    h_pred : tf.Tensor
        The predicted hydraulic head (GWL) sequence from the model.
        Expected shape is ``(batch_size, time_horizon, 1)``.
    time_steps : tf.Tensor
        The tensor of time values for the forecast horizon, used to
        calculate :math:`\\Delta t`. Must be broadcastable to the
        shape of ``s_pred`` and ``h_pred``. A common shape is
        ``(batch_size, time_horizon, 1)`` or ``(1, time_horizon, 1)``.
    C : Union[float, tf.Tensor]
        A learnable coefficient representing physical properties
        like compressibility (:math:`m_v`). Can be a scalar float,
        a tensor, or a trainable ``tf.Variable``.
    eps: float, default=1e-5
       Epsilon to prevent division by zero for static time
       
    Returns
    -------
    tf.Tensor
        A tensor representing the PDE residual at each interval of
        the forecast horizon. The shape will be
        ``(batch_size, time_horizon - 1, 1)``.

    Notes
    -----
    The underlying physical relationship is:

    .. math::

        \\frac{\\partial s}{\\partial t} =
        -m_v \\frac{\\partial \\sigma'}{\\partial t}

    where :math:`s` is subsidence, :math:`m_v` is the coefficient
    of volume compressibility, and :math:`\\sigma'` is the
    effective stress.

    Assuming total stress is constant and pore pressure
    :math:`u` is proportional to hydraulic head :math:`h`, this
    simplifies to:

    .. math::

        \\frac{\\partial s}{\\partial t} \\approx
        C \\cdot \\frac{\\partial h}{\\partial t}

    This function approximates the derivatives using a first-order
    finite difference scheme, making it suitable for sequence-to-sequence
    models:

    .. math::

        R = \\frac{s_{i+1} - s_i}{\\Delta t} + C \\cdot
        \\frac{h_{i+1} - h_i}{\\Delta t}

    The positive sign is used with the convention that a decrease in
    hydraulic head (a negative :math:`\\frac{\\partial h}{\\partial t}`)
    leads to an increase in subsidence (a positive
    :math:`\\frac{\\partial s}{\\partial t}`).

    Examples
    --------
    >>> import tensorflow as tf
    >>> from fusionlab.nn.pinn.op import compute_consolidation_residual
    >>> B, T, F = 4, 10, 1
    >>> s_sequence = tf.random.normal((B, T, F))
    >>> h_sequence = tf.random.normal((B, T, F))
    >>> # Time steps in years, for example
    >>> t_sequence = tf.reshape(tf.range(T, dtype=tf.float32), (1, T, 1))
    >>> C_coeff = tf.Variable(0.01, trainable=True)
    >>> residual = compute_consolidation_residual(
    ...     s_sequence, h_sequence, t_sequence, C_coeff
    ... )
    >>> print(f"Residual shape: {residual.shape}")
    Residual shape: (4, 9, 1)

    References
    ----------
    .. [1] Terzaghi, K., 1943. Theoretical Soil Mechanics.
           John Wiley and Sons, New York.

    """
    # Calculate time step intervals (Delta t)
    # delta_t shape: (batch_size, time_horizon - 1, 1)
    delta_t = time_steps[:, 1:, :] - time_steps[:, :-1, :]
    # Add a small epsilon to prevent division by zero for static time
    delta_t = delta_t + eps

    # Approximate derivatives using first-order finite differences
    # ds_dt shape: (batch_size, time_horizon - 1, 1)
    ds_dt = (s_pred[:, 1:, :] - s_pred[:, :-1, :]) / delta_t
    dh_dt = (h_pred[:, 1:, :] - h_pred[:, :-1, :]) / delta_t

    # --- Compute the PDE Residual ---
    # R = ds/dt + C * dh/dt
    # We expect this residual to be close to zero.
    pde_residual = ds_dt + C * dh_dt

    return pde_residual

@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG) 
def compute_gw_flow_residual(
    model: Model,
    coords: Dict[str, Tensor],
    K: Union[float, Tensor] = 1.0,
    Ss: Union[float, Tensor] = 1e-4,
    Q: Union[float, Tensor] = 0.0,
    h_pred: Optional[Tensor] = None
) -> Tensor:
    """
    Compute the residual of the 2D transient groundwater
    flow equation via PINN.

    The PDE residual is:

    .. math::
        R = K \left( \frac{\partial^2 h}{\partial x^2}
        + \frac{\partial^2 h}{\partial y^2} \right)
        + Q - S_s \frac{\partial h}{\partial t}

    Parameters
    ----------
    model : keras.Model
        Neural network predicting hydraulic head, h. It must
        accept a concatenated tensor of (t, x, y) inputs.
    coords : dict
        Dictionary with keys 't', 'x', 'y'. Each value is a
        tf.Tensor watched by GradientTape for differentiation.
    K : float or tf.Tensor, optional
        Hydraulic conductivity. Can be a trainable tf.Variable.
    Ss : float or tf.Tensor, optional
        Specific storage coefficient.
    Q : float or tf.Tensor, optional
        Source/sink term, e.g. recharge or pumping rate.
    h_pred : tf.Tensor, optional
        Precomputed model output. If None, compute via model.

    Returns
    -------
    tf.Tensor
        PDE residual at each collocation point, same shape as
        the model's h_pred output.

    Examples
    --------
    >>> from fusionlab.nn.pinn.op import compute_gw_flow_residual
    >>> # Assume `net` is a tf.keras.Model and t,x,y are tf.Variables
    >>> res = compute_gw_flow_residual(
    ...     model=net,
    ...     coords={'t': t, 'x': x, 'y': y},
    ...     K=0.5, Ss=1e-5, Q=0.1
    ... )
    >>> tf.reduce_mean(tf.square(res))
    <tf.Tensor: ...>

    References
    ----------
    [1] Bear, J. (1972). Dynamics of Fluids in Porous Media. Dover.
    [2] Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
        Physics-informed neural networks: A deep learning
        framework for solving forward and inverse problems
        involving nonlinear partial differential equations.
    """
    # Validate coords keys
    # if not all(k in coords for k in ('t', 'x', 'y')):
    #     raise ValueError(
    #         "coords must contain 't', 'x', and 'y' keys"
    #     )

    # t = coords['t']
    # x = coords['x']
    # y = coords['y']
    t, x, y = extract_txy_in(coords)

    # Persistent tape for first- and second-order grads
    # Use a persistent tape to compute multiple gradients
    with GradientTape(persistent=True) as tape:
        tape.watch((t, x, y))
        
        # Use a nested tape for second-order derivatives
        with GradientTape(persistent=True) as inner_tape:
            inner_tape.watch((t, x, y))
            if h_pred is None:
                inp = tf_concat([t, x, y], axis=1)
                h_pred = model(inp, training=True)
            # Ensure h_pred is watched by the inner tape 
            # if it was computed outside (though it's better
            # to compute it inside as shown above)
            inner_tape.watch(h_pred)

        # First-order derivatives
        dh_dt, dh_dx, dh_dy = inner_tape.gradient(
            h_pred, (t, x, y)
        )
        # Check for None gradients (can happen if an input 
        # is not in the computation graph)
        if dh_dt is None or dh_dx is None or dh_dy is None:
            raise ValueError(
                "Failed to compute one or more first-order gradients. "
                "Ensure t, x, y are inputs to the model and influence its "
                "hydraulic head prediction."
            )

    # Second-order spatial derivatives
    d2h_dx2 = tape.gradient(dh_dx, x)
    d2h_dy2 = tape.gradient(dh_dy, y)
    del tape, inner_tape
    if d2h_dx2 is None or d2h_dy2 is None:
        raise ValueError(
                "Failed to compute one or more second-order spatial gradients."
        )

    # Laplacian and residual
    # --- Compute the PDE Residual ---
    # R = K * Laplacian(h) + Q - Ss * dh/dt
    lap_h = d2h_dx2 + d2h_dy2
    residual = (K * lap_h) + Q - (Ss * dh_dt)
    return residual

def process_pinn_inputs(
    inputs: Union[Dict[str, Optional[Tensor]], List[Optional[Tensor]]],
    mode: str = 'as_dict',
    coord_keys: Tuple[str, str, str] = ('t', 'x', 'y'),
    coord_slice_map: Dict[str, int] = {'t': 0, 'x': 1, 'y': 2},
) -> Tuple[
    Tensor, Tensor, Tensor, Optional[Tensor], Tensor, Optional[Tensor]
]:
    """
    Processes and unpacks model inputs for PINN applications.

    This utility standardizes the handling of inputs for a PINN model,
    isolating the coordinate tensors required for differentiation from
    the feature tensors used for data-driven learning. It supports
    both dictionary and list-based input formats.

    Parameters
    ----------
    inputs : Union[Dict[str, Tensor], List[Optional[Tensor]]]
        The collection of input tensors for the model.

        - If ``mode='as_dict'`` (default), `inputs` should be a
          dictionary. It must contain the keys 'coords' and
          'dynamic_features'. Optional keys are 'static_features'
          and 'future_features'.
        - If ``mode='as_list'``, `inputs` should be a list or tuple.
          The expected order is:
          `[coords_tensor, dynamic_features, static_features, future_features]`
          where `static_features` and `future_features` are optional.

    mode : {'as_dict', 'as_list'}, default 'as_dict'
        Specifies the format of the `inputs` collection.

    coord_keys : Tuple[str, str, str], default ('t', 'x', 'y')
        A tuple defining the keys for the coordinate tensors to be
        returned. This parameter is currently for future compatibility
        and is not used in the logic.

    coord_slice_map : Dict[str, int], default {'t': 0, 'x': 1, 'y': 2}
        A dictionary mapping coordinate names to their integer index
        in the last dimension of the `coords` tensor. This defines how
        the `coords` tensor is sliced into individual coordinate tensors.

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Tensor, Optional[Tensor]]
        A tuple containing the unpacked tensors in the following order:
        ``(t, x, y, static_features, dynamic_features, future_features)``.
        `static_features` and `future_features` will be ``None`` if they
        were not provided in the input.

    Raises
    ------
    ValueError
        If an invalid `mode` is specified, or if required inputs for a
        given mode are missing.
    TypeError
        If `inputs` is not of the expected type for the specified `mode`.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from fusionlab.nn.pinn.op import process_pinn_inputs
    >>> # ---- Dictionary Mode ----
    >>> B, T, S_D, D_D, F_D = 4, 10, 2, 5, 3
    >>> inputs_dict = {
    ...     'coords': tf.random.normal((B, T, 3)), # t, x, y
    ...     'dynamic_features': tf.random.normal((B, T, D_D)),
    ...     'static_features': tf.random.normal((B, S_D)),
    ... }
    >>> t, x, y, s, d, f = process_pinn_inputs(inputs_dict, mode='as_dict')
    >>> print(f"t shape: {t.shape}, d shape: {d.shape}, f is None: {f is None}")
    t shape: (4, 10, 1), d shape: (4, 10, 5), f is None: True

    >>> # ---- List Mode ----
    >>> inputs_list = [
    ...     tf.random.normal((B, T, 3)), # coords
    ...     tf.random.normal((B, T, D_D)), # dynamic
    ...     tf.random.normal((B, S_D)), # static
    ... ]
    >>> t, x, y, s, d, f = process_pinn_inputs(inputs_list, mode='as_list')
    >>> print(f"s shape: {s.shape}, d shape: {d.shape}, f is None: {f is None}")
    s shape: (4, 2), d shape: (4, 10, 5), f is None: True
    """
    coords_tensor: Optional[Tensor] = None
    static_features: Optional[Tensor] = None
    dynamic_features: Optional[Tensor] = None
    future_features: Optional[Tensor] = None

    if mode == 'as_dict':
        if not isinstance(inputs, dict):
            raise TypeError(
                f"Expected `inputs` to be a dictionary for mode='as_dict',"
                f" but got {type(inputs)}."
            )
        # Required inputs for dictionary mode
        coords_tensor = inputs.get('coords')
        dynamic_features = inputs.get('dynamic_features')
        if coords_tensor is None or dynamic_features is None:
            raise ValueError(
                "For mode='as_dict', `inputs` must contain keys "
                "'coords' and 'dynamic_features'."
            )
        # Optional inputs
        static_features = inputs.get('static_features')
        future_features = inputs.get('future_features')

    elif mode == 'as_list':
        if not isinstance(inputs, (list, tuple)):
            raise TypeError(
                f"Expected `inputs` to be a list or tuple for "
                f"mode='as_list', but got {type(inputs)}."
            )
        num_inputs = len(inputs)
        if num_inputs < 2:
            raise ValueError(
                f"For mode='as_list', `inputs` must have at least 2 "
                f"elements: [coords, dynamic_features]. "
                f"Got {num_inputs} elements."
            )
        # Unpack based on the defined order
        coords_tensor = inputs[0]
        static_features = inputs[1] if num_inputs > 2 else None
        dynamic_features = inputs[2]
        future_features = inputs[3] if num_inputs > 3 else None

    else:
        raise ValueError(
            f"Invalid `mode`: '{mode}'. Must be 'as_dict' or 'as_list'."
        )

    # Slice the coordinates tensor to isolate t, x, and y
    # Keep the last dimension for concatenation or broadcasting later.
    t = coords_tensor[..., coord_slice_map['t']:coord_slice_map['t']+1]
    x = coords_tensor[..., coord_slice_map['x']:coord_slice_map['x']+1]
    y = coords_tensor[..., coord_slice_map['y']:coord_slice_map['y']+1]

    return (
        t, x, y,
        static_features,
        dynamic_features,
        future_features
    )

def calculate_gw_flow_pde_residual_from_derivs(
    dh_dt: Tensor,
    d2h_dx2: Tensor,
    d2h_dy2: Tensor,
    K: Union[float, Tensor],
    Ss: Union[float, Tensor],
    Q: Union[float, Tensor] = 0.0,
    name: Optional[str] = None
) -> Tensor:
    r"""
    Calculates the residual of the 2D transient groundwater flow equation
    using pre-computed derivatives.

    This function is intended to be used within a PINN framework where
    the derivatives of the hydraulic head :math:`h` with respect to time
    :math:`t` and spatial coordinates :math:`x, y` have already been
    computed (e.g., via automatic differentiation in the main model's
    forward pass).

    The implemented PDE is:

    .. math::

        S_s \frac{\partial h}{\partial t} = K \left( \frac{\partial^2 h}{\partial x^2} +
        \frac{\partial^2 h}{\partial y^2} \right) + Q

    The residual :math:`R` is therefore:

    .. math::

        R = K \left( \frac{\partial^2 h}{\partial x^2} +
        \frac{\partial^2 h}{\partial y^2} \right) + Q - S_s \frac{\partial h}{\partial t}

    Minimizing this residual (i.e., :math:`R \to 0`) at various
    spatio-temporal collocation points enforces the physical law.

    Parameters
    ----------
    dh_dt : tf.Tensor
        First partial derivative of hydraulic head with respect to time
        :math:`\left(\frac{\partial h}{\partial t}\right)`.
        Shape should match the points where the residual is evaluated.
    d2h_dx2 : tf.Tensor
        Second partial derivative of hydraulic head with respect to x
        :math:`\left(\frac{\partial^2 h}{\partial x^2}\right)`. Shape should match.
    d2h_dy2 : tf.Tensor
        Second partial derivative of hydraulic head with respect to y
        :math:`\left(\frac{\partial^2 h}{\partial y^2}\right)`. Shape should match.
    K : Union[float, tf.Tensor]
        Hydraulic conductivity. Can be a scalar float, a tf.Tensor (if
        spatially variable and predicted/provided), or a trainable
        ``tf.Variable``.
    Ss : Union[float, tf.Tensor]
        Specific storage. Can be a scalar float, a tf.Tensor, or a
        trainable ``tf.Variable``.
    Q : Union[float, tf.Tensor], default 0.0
        Source/sink term (e.g., recharge, pumping). Can be a scalar,
        a tf.Tensor, or a trainable ``tf.Variable``.
    name : str, optional
        Optional name for the TensorFlow operations.

    Returns
    -------
    tf.Tensor
        A tensor representing the PDE residual at each input point.
        Its shape will match the input derivative tensors.

    Notes
    -----
    - All input tensors (derivatives, K, Ss, Q) are expected to be
      broadcastable for element-wise operations.
    - This function assumes isotropic hydraulic conductivity :math:`K`.
      For anisotropic cases, the PDE and this function would need
      modification.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from fusionlab.nn.pinn.op import calculate_gw_flow_pde_residual_from_derivs
    >>> B, N_points = 4, 100 # Batch size, number of collocation points
    >>> dh_dt_vals = tf.random.normal((B, N_points, 1))
    >>> d2h_dx2_vals = tf.random.normal((B, N_points, 1))
    >>> d2h_dy2_vals = tf.random.normal((B, N_points, 1))
    >>> K_val = tf.constant(1.5e-4)
    >>> Ss_val = tf.constant(2.0e-5)
    >>> Q_val = tf.constant(0.0)
    >>> pde_residual = calculate_gw_flow_pde_residual_from_derivs(
    ...     dh_dt_vals, d2h_dx2_vals, d2h_dy2_vals, K_val, Ss_val, Q_val
    ... )
    >>> print(f"PDE Residual shape: {pde_residual.shape}")
    PDE Residual shape: (4, 100, 1)

    """
    if not KERAS_BACKEND:
        raise RuntimeError(
            "TensorFlow/Keras backend is required for this operation."
        )

    with tf_name_scope(name or "gw_flow_pde_residual"):
        # Ensure inputs are tensors
        K_tf = tf_constant(K, dtype=tf_float32) if isinstance(
            K, (float, int)) else K
        Ss_tf = tf_constant(Ss, dtype=tf_float32) if isinstance(
            Ss, (float, int)) else Ss
        Q_tf = tf_constant(Q, dtype=tf_float32) if isinstance(
            Q, (float, int)) else Q

        # Laplacian term: K * (d^2h/dx^2 + d^2h/dy^2)
        laplacian_h = d2h_dx2 + d2h_dy2
        diffusion_term = K_tf * laplacian_h

        # Storage term: Ss * dh/dt
        storage_term = Ss_tf * dh_dt
        
        # Source/sink term
        source_term = Q_tf
        
        # PDE Residual: R = Diffusion + Source - Storage
        pde_residual = diffusion_term + source_term - storage_term
        
    return pde_residual

def compute_gw_flow_derivatives(
    model_gwl_predictor_func: Callable[[Tensor, Tensor, Tensor], Tensor],
    t: Tensor,
    x: Tensor,
    y: Tensor
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    r"""
    Computes first and second order derivatives of predicted hydraulic
    head :math:`h` for the groundwater flow PDE using `tf.GradientTape`.

    This function is designed to be called within a context where `t`, `x`,
    and `y` are being watched by an outer `GradientTape` if necessary
    (e.g., if parameters of `model_gwl_predictor_func` are being trained).
    It primarily focuses on deriving :math:`\frac{\partial h}{\partial t}`,
    :math:`\frac{\partial^2 h}{\partial x^2}`, and
    :math:`\frac{\partial^2 h}{\partial y^2}`.

    Args:
        model_gwl_predictor_func (Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]):
            A callable function (e.g., a part of a Keras model or a
            lambda) that takes individual `t`, `x`, `y` coordinate tensors
            (each typically of shape `(Batch, N_points, 1)`) and returns
            the predicted hydraulic head `h_pred` (shape `(Batch, N_points, 1)`).
            This function must be differentiable with respect to `t`, `x`, `y`.
        t (tf.Tensor): Time coordinate tensor.
        x (tf.Tensor): Spatial x-coordinate tensor (e.g., longitude).
        y (tf.Tensor): Spatial y-coordinate tensor (e.g., latitude).

    Returns:
        Tuple[Optional[tf.Tensor], Optional[tf.Tensor], Optional[tf.Tensor]]:
            A tuple containing:
            - `dh_dt`: Derivative of h w.r.t. t.
            - `d2h_dx2`: Second derivative of h w.r.t. x.
            - `d2h_dy2`: Second derivative of h w.r.t. y.
            Returns (None, None, None) if any primary gradient is None.
    """
    if not KERAS_BACKEND:
        raise RuntimeError("TensorFlow/Keras backend required.")

    dh_dt, d2h_dx2, d2h_dy2 = None, None, None

    with GradientTape(persistent=True) as tape_h_derivs:
        tape_h_derivs.watch([t, x, y])
        
        # For second derivatives, need to compute first derivatives inside this tape
        with GradientTape(persistent=True) as tape_h_first_order:
            tape_h_first_order.watch([t, x, y])
            # Get h_pred from the specialized function/model part
            h_pred_for_derivs = model_gwl_predictor_func(t, x, y)
            # Ensure model output is watched if it's not directly from t,x,y
            # (though it should be for this to work)
            tape_h_first_order.watch(h_pred_for_derivs)

        # First order derivatives
        # We are interested in dh/dt for the PDE, and dh/dx, dh/dy for Laplacian
        dh_dt = tape_h_first_order.gradient(h_pred_for_derivs, t)
        dh_dx = tape_h_first_order.gradient(h_pred_for_derivs, x)
        dh_dy = tape_h_first_order.gradient(h_pred_for_derivs, y)

    if dh_dx is not None and dh_dy is not None:
        d2h_dx2 = tape_h_derivs.gradient(dh_dx, x)
        d2h_dy2 = tape_h_derivs.gradient(dh_dy, y)
    else: # If first order spatial derivatives are None, second order will be too
        logger.warning("Could not compute first-order spatial derivatives for GW flow PDE.")

    del tape_h_derivs
    del tape_h_first_order
    
    # Basic check for None gradients
    if dh_dt is None:
        logger.warning("dh/dt is None in compute_gw_flow_derivatives.")
    if d2h_dx2 is None:
        logger.warning("d2h/dx2 is None in compute_gw_flow_derivatives.")
    if d2h_dy2 is None:
        logger.warning("d2h/dy2 is None in compute_gw_flow_derivatives.")
        
    return dh_dt, d2h_dx2, d2h_dy2
