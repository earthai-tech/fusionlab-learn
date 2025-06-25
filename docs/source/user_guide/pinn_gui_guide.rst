.. _pinn_gui_guide:

=====================================
Subsidence PINN Mini GUI Guide
=====================================

This guide provides a complete walkthrough of the **Subsidence PINN
Mini GUI**, a desktop application designed to provide a user-friendly
interface for the complex forecasting workflows in ``fusionlab-learn``.

The application allows users who may not be familiar with Python to
load their own data, configure model parameters, run a full
training and forecasting pipeline, and view the results, all from a
simple graphical interface.

.. figure:: ../images/gui_overview.png
   :alt: Screenshot of Subsidence PINN Mini GUI
   :align: center
   :width: 80%

   A preview of the main application window with its detailed configuration panels.

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Launching the Application
-------------------------

The GUI is a tool within the ``fusionlab-learn`` library. To run it,
you must have the library and its dependencies (especially `PyQt5`)
installed.

1.  Navigate to the root directory of the `fusionlab-learn` project
    in your terminal.
2.  Run the application using the following command:

    .. code-block:: bash

       python -m fusionlab.tools.app.mini_forecaster_gui

    This will launch the main application window.


.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Prerequisites: Data Format Requirements
---------------------------------------

.. important::
   The **Subsidence PINN Mini GUI** is designed to work with a
   specific data structure. To ensure the workflow runs correctly,
   your uploaded CSV file **must** contain the following columns with these
   **exact names**:

   * ``longitude``: The spatial x-coordinate.
   * ``latitude``: The spatial y-coordinate.
   * ``year``: The time dimension column.
   * ``subsidence``: The primary target variable for land subsidence.
   * ``GWL``: The secondary target variable for Groundwater Level.

   The underlying PINN models (`TransFlowSubsNet` and `PIHALNet`) are
   specifically designed to model the coupled physical relationship
   between subsidence and groundwater levels. The workflow **will fail** if
   these two target columns are missing or named differently. For more 
   theoretical details, please see the :doc:`PINN Models guide
   </user_guide/models/pinn/index>`.

   **How to Fix Naming Issues:**
   If your dataset uses different names (e.g., `Lat`, `Lon`, `Date`), you
   **must** use the **"CSV Preview & Editing"** window that appears after
   loading your file to rename the columns to match the required names
   before running the workflow.

   **Feature Columns:**
   Similarly, any columns you specify in the **Feature Selection** panel
   (for Dynamic, Static, and Future features) must exist in your dataset.
   These should be provided as comma-separated lists.
   
.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

User Interface Guide
--------------------

The application is divided into several logical panels for configuration
and results.

**1. Data Input & Main Controls**
*********************************
These are the primary controls for managing the workflow.

* **Select CSV...**: Click this button to open a file dialog. Navigate
  to and select the `.csv` file containing your spatiotemporal data.
  The filename will appear next to the button upon successful selection.

* **City / Dataset**: This text field allows you to specify a name for
  your dataset (e.g., 'zhongshan', 'nansha'). This name is used
  internally to manage configurations and to create uniquely named
  output directories for saving results, preventing runs from
  overwriting each other.

* **Run**: Located at the bottom left, this button starts the
  end-to-end workflow using the current configuration. It becomes
  disabled while a process is running.
  
* **Reset**: Located at the top right, this button clears all logs and
  results and resets all configuration options to their default values.
  
* **Stop**: Appears next to *Reset* once the workflow is running.  
  Click to gracefully interrupt the current process (sequence generation,
  training, or forecasting). The button is disabled when the GUI is idle.
  
* **Inference**: This button becomes active only if a completed
  training run (a ``run_manifest.json`` file) is detected near the
  selected CSV file. Clicking it launches the inference workflow.
  
**2. Data Preview and Editing**
*******************************
After a CSV file is selected, a new **"Preview & Edit Data"** button
will appear. Clicking this opens a data preview window, allowing you to
perform basic cleaning and preparation steps directly within the GUI
before running the main workflow.

.. figure:: ../images/gui_data_editor.png
   :alt: Screenshot of the Data Preview and Editing window
   :align: center
   :width: 75%

   The data editor allows for quick modifications to the loaded dataset.

This window provides several useful tools:

* **Table Preview**: Displays the first several rows of your dataset,
  allowing you to verify that it was loaded correctly.
* **Delete row(s)**: Allows you to select and remove specific rows
  from the dataset.
* **Delete col(s)**: Allows you to select and remove unwanted columns.
* **Rename column**: Provides a dialog to rename a selected column.
* **Save / Apply**: Saves all changes you've made and closes the window,
  updating the dataset that will be used by the main workflow.
* **Cancel**: Closes the window without saving any changes.

**3. Model Configuration**
**************************
This panel allows you to configure the model's core architecture.

* **Architecture**: Choose between ``TransFlowSubsNet`` (the advanced,
  coupled-physics model) and ``PIHALNet`` (the consolidation-focused
  model).
* **Epochs**: Sets the maximum number of training epochs.
* **Batch Size**: Defines the number of samples processed in each batch
  during training.
* **Learning Rate**: Sets the initial learning rate for the Adam optimizer.
* **Model Type**: Sets the internal data handling mode, typically `'pihal'`
  or `'tft'`.
* **Attention Levels**: A comma-separated list defining which attention
  mechanisms to use (e.g., `'1, 2, 3'`).
* **Evaluate Coverage**: A checkbox to enable the calculation of quantile
  coverage score after prediction. 

**4. Training Parameters**
**************************
This panel controls the temporal aspects of the training and forecasting
process.

* **Train End Year**: The last year of data to be included in the training set.
* **Forecast Start Year**: The first year for which predictions will be made.
* **Forecast Horizon (Years)**: The number of years to predict into the future.
* **Time Steps (look-back)**: The number of historical time steps to use as
  input for the model's encoder.
* **Quantiles (comma-separated)**: A list of quantiles for probabilistic
  forecasting (e.g., `0.1, 0.5, 0.9`). Leave blank for point forecasting.
* **Checkpoint Format**: Select the file format used when saving model
  checkpoints—``weights`` (recommended for the GUI), ``keras``, or ``tf``.
  
**5. Physical Parameters**
**************************
This panel gives you fine-grained control over the physics-informed components.

* **Pinn Coeff C, K, Ss, Q**: For each physical parameter, you can select
  ``learnable`` to have the model infer its value, or provide a fixed
  numerical value.
* **λ Consolidation / λ GW Flow**: Sets the weights (:math:`\lambda_c`,
  :math:`\lambda_{gw}`) for the physics loss terms.
* **PDE Mode**: Controls which physics constraints are active during
  training (e.g., `'both'`, `'consolidation'`).
* **Weights (Subs. / GWL)**: Sets the relative importance of the data-fidelity
  loss for the two main targets (subsidence and groundwater level).

**6. Feature Selection**
************************
This panel allows you to specify which columns from your input data
should be used for the different feature streams.

* **Dyn. / Stat. / Future**: Enter the names of your columns, separated
  by commas, into the appropriate fields for **Dynamic**, **Static**,
  and **Future** features. Leaving a field as ``auto`` will let the
  application attempt to automatically detect the appropriate columns.
  
  
**7. Log and Output Panel**
***************************
The large text area at the bottom of the window is the Log Panel.
This is your primary window into the workflow's progress. It provides
real-time, timestamped feedback for each major step, from data loading
to model training and final visualization. Any warnings or errors that
occur during the process will be printed here, providing crucial
information for debugging.

Once the workflow is complete, this panel will also display the head of
the final results DataFrame and any generated plots, giving you an
immediate preview of the outcome.


**8. Viewing the Results**
**************************
Once the workflow finishes successfully, the GUI provides the results in
two main ways: status updates on the main window and an interactive plot viewer.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card::
      :shadow: none
      :class-card: no-border

      .. figure:: ../images/gui_results_view.png
         :alt: GUI after run completion
         :align: center

         **(a)** The main application window showing the 'Forecast finished'
         status and the final coverage score in the status bar.

   .. grid-item-card::
      :shadow: none
      :class-card: no-border

      .. figure:: ../images/gui_plot_viewer.png
         :alt: Plot visualization window
         :align: center

         **(b)** The interactive plot viewer window that opens automatically
         to display all generated visualizations.

* **Main Window Updates (a):** A checkmark and "Forecast finished"
  message appear at the top.If the **"Evaluate Coverage"** checkbox in the
  Model Configuration panel was ticked, the calculated coverage score
  (e.g., `cov-result: 0.792`) will be displayed in the bottom status bar.

* **Interactive Plot Viewer (b):** A new window opens to display all
  plots generated during the run, such as the training history and
  forecast visualizations. This viewer allows you to inspect the
  visuals closely and provides options to "Save as..." or "Copy to
  clipboard" for easy export.

* **Zoom & Pan Controls:** The viewer includes a translucent floating
  toolbar in the upper-left corner with **“+”** (zoom in),
  **“–”** (zoom out) and **“□”** (fit view) buttons.
  You can also scroll the mouse wheel to zoom and drag with the left
  mouse button to pan the image for detailed inspection.
  
* **Final Log Messages**: The log panel will show the final messages,
  including confirmation that all figures have been saved and the path
  to the final output directory.
  
**9. Saving Results and Artifacts**
************************************
Upon successful completion of a run, the application automatically saves all
generated artifacts and plots to a dedicated output directory. This ensures that your
configuration, processed data, trained model, and results are preserved
for later analysis and reproducibility.

The output directory is structured using the parameters from your
configuration: ``results_pinn/<city_name>_<model_name>_run/``

Inside this directory, you will find:

* **Processed Data:** Intermediate CSV files from the preprocessing steps.
* **Fitted Scalers:** The saved scikit-learn scalers and encoders as
  `.joblib` files.
* **Trained Model:** The best model checkpoint saved in the `.keras` format.
* **Forecast DataFrame:** The final prediction results in a `.csv` file.
* **Visualizations:** All generated plots (e.g., training history, forecast
  maps) saved as `.png` and `.pdf` files.
* **Coverage Results:** If ``Evaluate Coverage`` is enabled, the coverage
  score results will also be included in the output.
  
.. _pinn_gui_inference_guide:

Running Inference with a Trained Model
----------------------------------------
The GUI is not just for training; it's also a powerful tool for
running **inference**. This allows you to take a model you have
already trained and apply it to a new, unseen dataset to generate
predictions.

The inference workflow is automatically enabled when the application
detects that a model has already been trained.

.. figure:: ../images/gui_inference_enabled.png
   :alt: Screenshot of the GUI with the Inference button enabled.
   :align: center
   :width: 80%

   After a training run is found, the "Inference" button becomes
   active, allowing you to run predictions with the trained model.

**How it Works:**

1.  **Automatic Detection:** When you select a CSV file using the
    **"Select CSV..."** button, the application automatically searches
    the surrounding directories for a ``run_manifest.json`` file. This
    file, created at the end of a successful training run, contains all
    the information about the trained model and its artifacts.

2.  **Enabling the "Inference" Button:** If a manifest file is found,
    the **"Inference"** button at the top right of the window will
    become active and turn blue, as shown in the screenshot above. Its
    tooltip will confirm that a trained model has been detected.

3.  **Launching the Inference Workflow:**

    * Click the **"Inference"** button.
    * You will be prompted to select a **new CSV file** containing the
      data you want to run predictions on. This should be a file with
      the same structure as your original training data.
    * The application will then use the ``PredictionPipeline`` to:
    
      * Load the pre-trained model and its specific scalers/encoders.
      * Process your new data using these loaded artifacts.
      * Generate a forecast.
      * Display the results and visualizations in the output panel.

This workflow provides a seamless way to apply your trained models to
new data without having to re-run the entire training process.