.. _geoprior_v3_train_tab:

Train tab
=========

.. figure:: /_static/apps/geoprior_v3/04_train_tab.png
   :alt: Train tab tab.
   :width: 96%

   Train tab tab layout (example screenshot).

Purpose
-------

Train a GeoPrior model (Stage-2). Includes physics weights, optimizer settings, epochs, batch size, and evaluation toggles.

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

Trained model checkpoints, training logs, metrics CSV/JSON, and optional diagnostic plots.

Troubleshooting
---------------

Add the most common error messages for this tab and what they mean.
Link to :doc:`../reference/troubleshooting` when the issue is generic.
