.. _glossary:

==========
Glossary
==========

This glossary defines key terms and concepts used in the
``fusionlab`` library and the domain of time series forecasting.

.. glossary::
   :sorted:

   ACF (Autocorrelation Function)
       A function that measures the correlation between a time series
       and lagged versions of itself. Used to identify seasonality and
       autoregressive patterns. See :func:`~fusionlab.utils.ts_utils.ts_corr_analysis`.

   ADF Test (Augmented Dickey-Fuller Test)
       A statistical test used to check for stationarity in a time
       series. The null hypothesis is that the series has a unit root
       (is non-stationary). See :func:`~fusionlab.utils.ts_utils.trend_analysis`.

   Additive Decomposition
       A model for time series decomposition where the components are
       summed: $Y_t = \text{Trend}_t + \text{Seasonal}_t + \text{Residual}_t$.
       See :func:`~fusionlab.utils.ts_utils.decompose_ts`.

   Anomaly Detection
       The process of identifying data points, events, or patterns
       that deviate significantly from the expected or normal behavior
       in a dataset. :class:`~fusionlab.nn.XTFT` includes integrated
       strategies for this.

   Anomaly Score
       A numerical value indicating the degree to which a data point
       is considered anomalous. Higher scores typically represent
       greater abnormality. See :func:`~fusionlab.nn.utils.compute_anomaly_scores`.

   Attention Mechanism
       A technique in neural networks that allows the model to
       dynamically weigh the importance of different parts of the
       input data (e.g., different time steps or features) when
       making predictions or creating representations. Includes
       variants like Self-Attention, Cross-Attention, Multi-Head
       Attention. See :ref:`Attention Mechanisms <user_guide_components>`.

   Autocorrelation
       The correlation of a time series with lagged versions of itself.
       See :term:`ACF (Autocorrelation Function)`.

   Autoencoder
       A type of neural network trained to reconstruct its input. Often
       used for dimensionality reduction or anomaly detection, where high
       reconstruction error can indicate an anomaly. See
       :class:`~fusionlab.nn.anomaly_detection.LSTMAutoencoderAnomaly`.

   Backtesting
       A method for evaluating a forecasting model's performance on
       historical data by simulating how the model would have performed
       if it had been used in the past. Often involves rolling or expanding
       windows. See :term:`Time Series Cross-Validation`.

   Batch Size
       The number of samples processed together in one iteration during
       model training or inference.

   CLI (Command-Line Interface)
       A text-based interface used to run programs or scripts from a
       terminal or command prompt. See :doc:`/user_guide/tools`.

   Coverage Score
       A metric used to evaluate probabilistic forecasts (specifically
       prediction intervals). It measures the proportion of actual
       values that fall within the predicted interval (e.g., between
       the 10th and 90th percentiles).

   Cross-Attention
       A type of attention mechanism where one sequence (query) attends
       to a different sequence (key/value), allowing interaction
       between distinct inputs. See
       :class:`~fusionlab.nn.components.CrossAttention`.

   Cross-Validation (CV)
       See :term:`Time Series Cross-Validation`.

   Decomposition
       The process of breaking down a time series into its underlying
       components, typically Trend, Seasonality, and Residuals.
       See :func:`~fusionlab.utils.ts_utils.decompose_ts`.

   Detrending
       The process of removing the trend component from a time series
       to isolate other patterns like seasonality or residuals.
       See :func:`~fusionlab.utils.ts_utils.transform_stationarity`.

   Differencing
       A transformation applied to time series data to stabilize the
       mean by subtracting previous observations from current ones
       ($Y'_t = Y_t - Y_{t-d}$). Used to achieve :term:`Stationarity`.
       See :func:`~fusionlab.utils.ts_utils.transform_stationarity`.

   Dynamic Features
       Features whose values change over time within a sequence (e.g.,
       past sales, temperature, promotions that occurred). Input to the
       temporal processing parts of models like TFT/XTFT.

   Early Stopping
       A regularization technique used during model training where
       training is stopped early if performance on a validation set
       stops improving (or starts degrading) for a certain number of
       epochs (`patience`).

   Embedding
       A learned, typically lower-dimensional, dense vector
       representation of discrete variables (like categorical features)
       or continuous features. Models like TFT/XTFT use embeddings for
       various inputs. See
       :class:`~fusionlab.nn.components.MultiModalEmbedding`.

   Epoch
       One complete pass through the entire training dataset during
       model training.

   Exogenous Variables
       External variables that can influence the target variable but are
       not directly influenced by it within the model's scope (e.g.,
       weather affecting sales, but sales not affecting weather). Often
       used as :term:`Future Features` if known in advance.

   Forecast Horizon
       The number of future time steps for which predictions are generated.
       Also referred to as `H`.

   Fourier Features / Transform
       Features derived from the Discrete Fourier Transform (DFT or FFT)
       of a time series. They represent the magnitude (or phase) of
       different frequency components and can capture complex
       periodicities. See :func:`~fusionlab.utils.ts_utils.ts_engineering`.

   Future Features (Known Covariates)
       Features whose values are known or can be reliably estimated for
       future time steps at the time of prediction (e.g., upcoming
       holidays, planned promotions, day of the week). TFT/XTFT are
       designed to leverage this information.

   GRN (Gated Residual Network)
       A core building block in TFT/XTFT, consisting of dense layers,
       gating mechanisms (like GLU), non-linear activation, and a
       residual connection. Used for flexible feature transformations.
       See :class:`~fusionlab.nn.components.GatedResidualNetwork`.

   GLU (Gated Linear Unit)
       A gating mechanism often used within GRNs, calculated as
       $a \odot \sigma(b)$, where $a$ and $b$ are typically outputs of
       linear layers, $\odot$ is element-wise multiplication, and $\sigma$
       is the sigmoid function.

   Heuristic
       A practical approach or rule of thumb used for problem-solving
       or decision-making, often based on experience or simplified logic,
       especially when an optimal solution is complex to find (e.g.,
       heuristic methods for choosing decomposition type).

   Hyperparameter
       A parameter whose value is set *before* the learning process begins,
       controlling the model architecture or training algorithm (e.g.,
       learning rate, number of hidden units, dropout rate). Contrast with
       model parameters learned during training (e.g., weights).

   Hyperparameter Tuning / Optimization
       The process of systematically searching for the optimal set of
       hyperparameters for a model to achieve the best performance on a
       given task or dataset. See :doc:`/user_guide/forecast_tuner`.

   IQR (Interquartile Range)
       A measure of statistical dispersion, calculated as the difference
       between the 75th percentile (Q3) and the 25th percentile (Q1).
       Used in outlier detection. See
       :func:`~fusionlab.utils.ts_utils.ts_outlier_detector`.

   Keras Tuner
       A library for automating hyperparameter tuning for Keras models,
       used by ``fusionlab.nn.forecast_tuner``.

   KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin Test)
       A statistical test used to check for stationarity in a time
       series. The null hypothesis is that the series is stationary
       around a deterministic trend (level or linear). See
       :func:`~fusionlab.utils.ts_utils.trend_analysis`.

   Lag Features
       Features created by shifting a time series back by one or more
       time steps ($X_{t-k}$). Allows models to use past values as
       predictors. See :func:`~fusionlab.utils.ts_utils.create_lag_features`.

   Latent Space / Representation
       A lower-dimensional space into which high-dimensional data is
       encoded, typically capturing the most salient features. Used in
       autoencoders. See
       :class:`~fusionlab.nn.anomaly_detection.LSTMAutoencoderAnomaly`.

   Layer Normalization
       A normalization technique applied across the features dimension
       for a single data sample, often used in Transformer-based models
       and GRNs to stabilize training.

   Lookback Period / Window
       The number of past time steps used as input features to predict
       future values. Corresponds to `sequence_length` or `time_steps`
       parameters.

   LOESS (Locally Estimated Scatterplot Smoothing)
       A non-parametric regression method used to fit a smooth curve
       through data points. Used internally by the :term:`STL`
       decomposition method.

   LSTM (Long Short-Term Memory)
       A type of Recurrent Neural Network (RNN) architecture capable of
       learning long-range dependencies in sequential data, often used
       as encoders in time series models. See
       :class:`~fusionlab.nn.components.MultiScaleLSTM`.

   MAE (Mean Absolute Error)
       A metric for evaluating regression models, calculated as the
       average of the absolute differences between predicted and actual
       values.

   MSE (Mean Squared Error)
       A common loss function and metric for regression, calculated as
       the average of the squared differences between predicted and
       actual values.

   Multi-Head Attention
       An extension of the basic attention mechanism where attention is
       calculated multiple times in parallel with different learned linear
       projections (heads). The results are concatenated and projected,
       allowing the model to jointly attend to information from different
       representation subspaces. See :ref:`Attention Mechanisms <user_guide_components>`.

   Multi-Horizon Forecasting
       Predicting multiple time steps into the future simultaneously, rather
       than just the single next step. `forecast_horizon` > 1.

   Multi-Scale Processing
       Analyzing a time series at different temporal resolutions or
       frequencies simultaneously, e.g., using LSTMs on daily and weekly
       sampled data. See :class:`~fusionlab.nn.components.MultiScaleLSTM`.

   Multiplicative Decomposition
       A model for time series decomposition where the components are
       multiplied: $Y_t = \text{Trend}_t \times \text{Seasonal}_t \times \text{Residual}_t$.
       Often suitable for series where seasonal variation or noise scales
       with the trend level. See :func:`~fusionlab.utils.ts_utils.decompose_ts`.

   NumPy Style Docstrings
       A convention for formatting Python docstrings, characterized by
       sections like Parameters, Returns, Examples, etc. Used by `fusionlab`.
       See :ext:`sphinx.ext.napoleon`.

   One-Hot Encoding
       A process of converting categorical integer features into a binary
       vector format where only one element is 'hot' (1) and the rest are 0.

   Outlier
       A data point that differs significantly from other observations.
       See :func:`~fusionlab.utils.ts_utils.ts_outlier_detector`.

   PACF (Partial Autocorrelation Function)
       Measures the correlation between a time series and its lag, after
       removing the linear dependence on shorter lags. Used to identify
       autoregressive order. See :func:`~fusionlab.utils.ts_utils.ts_corr_analysis`.

   Pinball Loss
       See :term:`Quantile Loss`.

   Point Forecast
       A single value prediction for each future time step, typically
       representing the expected value or median of the target distribution.
       Contrast with :term:`Quantile Forecast`.

   Positional Encoding
       A technique used primarily in Transformer-based models to inject
       information about the position or order of elements in a sequence,
       as self-attention mechanisms are otherwise permutation-invariant.
       See :class:`~fusionlab.nn.components.PositionalEncoding`.

   Probabilistic Forecasting
       Forecasting that provides an estimate of the uncertainty associated
       with predictions, typically by outputting a full predictive
       distribution or specific quantiles. See :term:`Quantile Forecast`.

   Quantile
       A point below which a certain percentage of the data falls. For
       example, the 0.1 quantile (or 10th percentile) is the value below
       which 10% of the data lies.

   Quantile Forecast
       A type of probabilistic forecast where the model predicts specific
       quantiles (e.g., 0.1, 0.5, 0.9) of the target variable's future
       distribution. This allows constructing prediction intervals.

   Quantile Loss (Pinball Loss)
       A loss function used for training models to predict specific
       quantiles. It penalizes errors asymmetrically based on the target
       quantile. See :func:`~fusionlab.nn.losses.combined_quantile_loss`.

   R² Score (Coefficient of Determination)
       A statistical measure representing the proportion of the variance
       in the dependent variable that is predictable from the independent
       variables. Ranges from -inf to 1.

   Residual
       The difference between the observed value and the value predicted
       or fitted by a model. In decomposition, it's the component left
       after removing trend and seasonality.

   Rolling Statistics / Window
       Statistics (e.g., mean, standard deviation) calculated over a
       sliding window of fixed size moving through the time series. Used
       to visualize local trends or volatility. See
       :func:`~fusionlab.utils.ts_utils.ts_engineering`.

   Scaler
       A data preprocessing tool (e.g., `StandardScaler`, `MinMaxScaler`
       from scikit-learn) used to normalize or scale numerical features,
       often necessary for optimal neural network training.

   Scaling
       The process of transforming numerical features to a standard range
       (e.g., [0, 1] for MinMaxScaler) or distribution (e.g., mean 0,
       std dev 1 for StandardScaler).

   SDT (Seasonal Decomposition of Time series)
       The classical method for time series decomposition available in
       `statsmodels`, supporting additive and multiplicative models.
       See :func:`~fusionlab.utils.ts_utils.decompose_ts`.

   Seasonality
       Patterns in a time series that repeat over a fixed period (e.g.,
       daily, weekly, yearly).

   Self-Attention
       An attention mechanism where a sequence attends to itself, allowing
       different positions within the sequence to interact and weigh each
       other's importance.

   Sequence Length
       See :term:`Lookback Period / Window`.

   Sequence-to-Sequence (Seq2Seq) Model
       A type of neural network architecture that maps an input sequence
       to an output sequence, commonly used in machine translation, text
       summarization, and time series forecasting.

   StandardScaler
       A preprocessing technique from scikit-learn that standardizes
       features by removing the mean and scaling to unit variance (Z-score).

   Static Features
       Features associated with a time series that do not change over
       time (e.g., store location ID, product category, sensor type).
       TFT/XTFT can leverage these as context.

   Stationarity
       A property of a time series whose statistical properties (like mean,
       variance, autocorrelation) are constant over time. Many classical
       time series models assume stationarity.

   STL (Seasonal-Trend decomposition using LOESS)
       A robust method for decomposing a time series into trend, seasonal,
       and residual components, available in `statsmodels`. See
       :func:`~fusionlab.utils.ts_utils.decompose_ts`.

   Supervised Learning
       A type of machine learning where the model learns a mapping from
       input features to output labels based on labeled training examples.
       Time series forecasting is often framed as a supervised task where
       past data predicts future data.

   TensorFlow
       An open-source machine learning framework developed by Google, used
       as the primary backend for ``fusionlab``'s neural network models.

   TFT (Temporal Fusion Transformer)
       A powerful deep learning architecture specifically designed for
       multi-horizon time series forecasting, capable of handling diverse
       feature types and providing interpretable outputs. See
       :class:`~fusionlab.nn.TemporalFusionTransformer`.

   Time Series
       A sequence of data points indexed (or graphed) in time order.

   Time Series Cross-Validation
       A cross-validation strategy for time series data that respects
       temporal order. Typically involves training on past data and
       testing on future data, often using expanding or rolling windows.
       See :func:`~fusionlab.utils.ts_utils.ts_split` (using `split_type='cv'`).

   Trend
       The long-term increase or decrease in a time series, ignoring
       short-term fluctuations and seasonality.

   Univariate Time Series
       A time series consisting of observations on only a single variable
       over time.

   Multivariate Time Series
       A time series consisting of observations on multiple variables
       over time.

   VSN (Variable Selection Network)
       A component within TFT/XTFT that learns to assign importance weights
       to different input features, aiding interpretability and potentially
       improving performance by focusing on relevant inputs. See
       :class:`~fusionlab.nn.components.VariableSelectionNetwork`.

   XTFT (Extreme Temporal Fusion Transformer)
       An enhanced version of TFT developed within ``fusionlab`` (or based
       on related research), incorporating more advanced components like
       multi-scale LSTMs, specialized attention mechanisms, and integrated
       anomaly detection. See :class:`~fusionlab.nn.XTFT`.

   Z-Score
       A statistical measurement describing a value's relationship to the
       mean of a group of values, measured in terms of standard deviations.
       Used in :func:`~fusionlab.utils.ts_utils.ts_outlier_detector`.