
# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

from __future__ import annotations
from typing import Optional, Dict, Any

import numpy as np

from ..._fusionlog import fusionlog, OncePerMessageFilter
from .. import KERAS_DEPS, dependency_message
from ._geoprior_subnet import GeoPriorSubsNet

register_keras_serializable = KERAS_DEPS.register_keras_serializable

logger = fusionlog().get_fusionlab_logger(__name__)
logger.addFilter(OncePerMessageFilter())

DEP_MSG = dependency_message("nn.pinn.models")

__all__ = ["PoroElasticSubsNet"]


@register_keras_serializable("fusionlab.nn.pinn", name="PoroElasticSubsNet")
class PoroElasticSubsNet(GeoPriorSubsNet):
    """
    Poroelastic surrogate variant of GeoPriorSubsNet.

    Same architecture and outputs as GeoPriorSubsNet, but with:

    * default ``pde_mode='consolidation'`` (no groundwater-flow residual);
    * effective drained thickness enabled
      (``use_effective_h=True``, ``hd_factor < 1``);
    * stronger geomechanical consistency prior and soft bounds on
      (H, K, S_s) via larger default lambda weights.

    Intended as a physics-driven baseline for ablation / comparison.
    """

    def __init__(
        self,
        static_input_dim: int,
        dynamic_input_dim: int,
        future_input_dim: int,
        # keep all public kwargs, but change some defaults:
        pde_mode: str = "consolidation",
        use_effective_h: bool = True,
        hd_factor: float = 0.6,
        kappa_mode: str = "bar",
        scale_pde_residuals: bool = True,
        scaling_kwargs: Optional[Dict[str, Any]] = None,
        name: str = "PoroElasticSubsNet",
        **kwargs,
    ):
        # ------------------------------------------------------------------
        # 1) Merge scaling_kwargs with default bounds, if not provided.
        # ------------------------------------------------------------------
        if scaling_kwargs is None:
            scaling_kwargs = {}

        bounds = dict(scaling_kwargs.get("bounds", {}) or {})

        # Only fill missing keys; do not overwrite user-provided ones.
        default_bounds = dict(
            H_min=5.0,
            H_max=80.0,
            logK_min=float(np.log(1e-8)),
            logK_max=float(np.log(1e-3)),
            logSs_min=float(np.log(1e-7)),
            logSs_max=float(np.log(1e-3)),
        )
        for k, v in default_bounds.items():
            bounds.setdefault(k, v)

        scaling_kwargs["bounds"] = bounds

        logger.info(
            "Initializing GeoPriorStrongPrior with "
            f"pde_mode={pde_mode}, use_effective_h={use_effective_h}, "
            f"hd_factor={hd_factor}, kappa_mode={kappa_mode}, "
            f"bounds={bounds}"
        )

        super().__init__(
            static_input_dim=static_input_dim,
            dynamic_input_dim=dynamic_input_dim,
            future_input_dim=future_input_dim,
            # pass through everything else, with updated defaults:
            pde_mode=pde_mode,
            use_effective_h=use_effective_h,
            hd_factor=hd_factor,
            kappa_mode=kappa_mode,
            scale_pde_residuals=scale_pde_residuals,
            scaling_kwargs=scaling_kwargs,
            name=name,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Stronger default physics weights in compile()
    # ------------------------------------------------------------------
    def compile(
        self,
        lambda_cons: float = 1.0,
        lambda_gw: float = 0.0,   # gw_flow off by default for surrogate
        lambda_prior: float = 5.0,
        lambda_smooth: float = 1.0,
        lambda_mv: float = 0.1,
        lambda_bounds: float = 0.05,
        mv_lr_mult: float = 0.5,
        kappa_lr_mult: float = 0.5,
        **kwargs,
    ):
        """
        Compile with stronger defaults for the geomechanical prior.

        Compared to GeoPriorSubsNet, this variant:

        * sets ``lambda_gw=0.0`` (no groundwater-flow residual),
        * increases ``lambda_prior`` and ``lambda_bounds`` so that
          :math:`tau` is tightly tied to :math:`tau_phys`,
        * gives :math:`m_v` and :math:`kappa` a smaller LR multiplier
          so they move more conservatively.
        """
        logger.info(
            "Compiling PoroElasticSubsNet with "
            f"lambda_cons={lambda_cons}, lambda_gw={lambda_gw}, "
            f"lambda_prior={lambda_prior}, lambda_smooth={lambda_smooth}, "
            f"lambda_mv={lambda_mv}, lambda_bounds={lambda_bounds}"
        )
        return super().compile(
            lambda_cons=lambda_cons,
            lambda_gw=lambda_gw,
            lambda_prior=lambda_prior,
            lambda_smooth=lambda_smooth,
            lambda_mv=lambda_mv,
            lambda_bounds=lambda_bounds,
            mv_lr_mult=mv_lr_mult,
            kappa_lr_mult=kappa_lr_mult,
            **kwargs,
        )
