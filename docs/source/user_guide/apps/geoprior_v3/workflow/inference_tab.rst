.. _geoprior_v3_inference_tab:

Inference tab
=============

.. figure:: /_static/apps/geoprior_v3/06_inference_tab.png
   :alt: Inference tab tab.
   :width: 96%

   Inference tab tab layout (example screenshot).

Purpose
-------

Generate forecasts using a trained model, export tables, and optionally run evaluation on held-out splits.

Typical flow
------------

1. Choose or confirm inputs for this tab.
2. Adjust parameters (they are store-backed and persist).
3. Click **Run** and watch progress in the log panel.
4. Review outputs in the **Results** tab.

Key controls
------------

Document the important controls in this tab. Prefer describing them in terms of:

- **What it changes** (store key / config field)
- **Why you would change it**
- **Safe defaults**

Outputs
-------

Forecast CSV/NPZ exports, reliability/sharpness summaries, and inspection-ready artifacts.

Troubleshooting
---------------

Add the most common error messages for this tab and what they mean.
Link to :doc:`../reference/troubleshooting` when the issue is generic.
