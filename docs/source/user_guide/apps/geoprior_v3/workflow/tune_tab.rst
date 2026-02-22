.. _geoprior_v3_tune_tab:

Tune tab
========

.. figure:: /_static/apps/geoprior_v3/05_tune_tab.png
   :alt: Tune tab tab.
   :width: 96%

   Tune tab tab layout (example screenshot).

Purpose
-------

Run hyperparameter search over selected knobs. Uses the same training pipeline but manages multiple trials and writes a tuning manifest.

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

Tuning trial summaries, best-config manifest, and the best model checkpoint.

Troubleshooting
---------------

Add the most common error messages for this tab and what they mean.
Link to :doc:`../reference/troubleshooting` when the issue is generic.
