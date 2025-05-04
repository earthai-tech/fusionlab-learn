.. _glossary:

=========
Glossary
=========

This glossary defines key terms, abbreviations, and concepts used
throughout the ``fusionlab-learn`` library documentation and the broader
domain of time series forecasting and deep learning.

.. glossary::
   :sorted:

   ACF (Autocorrelation Function)
       A function measuring the correlation between a time series
       and lagged versions of itself. Used to identify seasonality,
       memory, and autoregressive patterns. See
       :func:`~fusionlab.utils.ts_utils.ts_corr_analysis`.

   ADF Test (Augmented Dickey-Fuller Test)
       A statistical test for :term:`Stationarity`. The null
       hypothesis assumes the series has a unit root (is
       non-stationary). A low p-value suggests stationarity. See
       :func:`~fusionlab.utils.ts_utils.trend_analysis`.

   Additive Decomposition
       A model for time series decomposition where the components are
       summed: :math:`Y_t = Trend_t + Seasonal_t + Residual_t`.
       Often suitable when seasonality/residuals do not scale with the
       trend. See :func:`~fusionlab.utils.ts_utils.decompose_ts`.

   Anomaly Detection
       The process of identifying data points, events, or patterns
       that deviate significantly from expected or normal behavior.
       See the :doc:`Anomaly Detection guide </user_guide/anomaly_detection>`.

   Anomaly Score
       A numerical value indicating the degree to which a data point
       or sequence is considered anomalous. Higher scores typically
       represent greater abnormality. See
       :func:`~fusionlab.nn.utils.compute_anomaly_scores`.

   Attention Mechanism
       A technique allowing neural networks to dynamically weigh the
       importance of different input parts (e.g., time steps, features)
       when forming representations or predictions. Includes variants
       like Self-Attention, Cross-Attention, Multi-Head Attention.
       See the :ref:`Attention Mechanisms <user_guide_components>` section.

   Autocorrelation
       The correlation of a time series with lagged versions of itself.
       See :term:`ACF (Autocorrelation Function)`.

   Autoencoder
       A neural network trained for unsupervised reconstruction of its
       input, often via a compressed :term:`Latent Space`. High
       reconstruction error can indicate anomalies. See
       :class:`~fusionlab.nn.anomaly_detection.LSTMAutoencoderAnomaly`.

   Autoregressive Model (AR Model)
       A time series model where the current value is predicted based
       on a linear combination of its own past values (lags).

   Backtesting
       Evaluating a forecasting model's performance on historical data
       by simulating its past usage, often with rolling or expanding
       windows to respect temporal order. See
       :term:`Time Series Cross-Validation`.

   Batch Normalization
       A normalization technique applied across the batch dimension,
       typically after a linear layer and before activation, to stabilize
       training by normalizing layer inputs. Compare with
       :term:`Layer Normalization`.

   Batch Size
       The number of samples processed together in one forward/backward
       pass during model training or inference.

   Bunch
       A utility class (similar to a dictionary but allowing attribute-style
       access) used by ``fusionlab-learn`` dataset loaders to return data and
       metadata. See :class:`~fusionlab.api.bunch.Bunch`.

   Categorical Feature
       A feature whose values belong to a finite set of discrete
       categories (e.g., 'product_type', 'day_of_week'). Often requires
       :term:`One-Hot Encoding` or :term:`Embedding` before use in models.

   CLI (Command-Line Interface)
       A text-based interface for interacting with software via commands
       entered in a terminal. See :doc:`/user_guide/tools`.

   Continuous Feature
       A feature whose values can take any number within a range (e.g.,
       'temperature', 'sales_amount'). Often requires :term:`Scaling`.

   Coverage Score
       A metric evaluating probabilistic forecasts (prediction intervals).
       It measures the proportion of actual values falling within the
       predicted interval (e.g., between the 0.1 and 0.9 quantiles).

   Cross-Attention
       Attention where one sequence (query) attends to a *different*
       sequence (key/value), modeling interactions between distinct inputs.
       See :class:`~fusionlab.nn.components.CrossAttention`.

   Cross-Validation (CV)
       See :term:`Time Series Cross-Validation`.

   Decomposition
       Separating a time series into constituent components like Trend,
       Seasonality, and Residuals. See
       :func:`~fusionlab.utils.ts_utils.decompose_ts`.

   Detrending
       Removing the trend component from a time series. See
       :func:`~fusionlab.utils.ts_utils.transform_stationarity`.

   Differencing
       Transforming a time series by subtracting previous values, e.g.,
       :math:`Y'_t = Y_t - Y_{t-d}`. Used to achieve :term:`Stationarity`.
       See :func:`~fusionlab.utils.ts_utils.transform_stationarity`.

   Dynamic Features
       Features whose values change over time (e.g., past sales, weather,
       day-of-week). Used as historical inputs in TFT/XTFT.

   Dynamic Time Window
       A component that adaptively selects or weights the most recent
       time steps from a sequence. See
       :class:`~fusionlab.nn.components.DynamicTimeWindow`.

   Early Stopping
       A regularization technique stopping training early if validation
       performance plateaus or degrades, preventing overfitting.

   Embedding
       A learned, dense vector representation, typically of lower
       dimension, used to represent discrete inputs (like categories) or
       project continuous inputs. See
       :class:`~fusionlab.nn.components.MultiModalEmbedding`.

   Epoch
       One complete pass through the entire training dataset.

   Exogenous Variables
       External variables influencing the target variable but not
       influenced by it (e.g., weather affecting sales). Often used as
       :term:`Future Features` if known in advance.

   Feature Engineering
       The process of creating new input features from raw data to
       improve model performance. See
       :func:`~fusionlab.utils.ts_utils.ts_engineering`.

   Forecast Horizon
       The number of future time steps (:math:`H`) for which predictions
       are generated.

   Fourier Features / Transform
       Features derived from the Discrete Fourier Transform (DFT or FFT),
       representing the magnitude/phase of different frequency components.
       Useful for capturing complex periodicities. See
       :func:`~fusionlab.utils.ts_utils.ts_engineering`.

   Future Features (Known Covariates)
       Features whose values are known in advance for future time steps
       at the time of prediction (e.g., holidays, promotions, day-of-week).
       Leveraged by TFT/XTFT.

   Gate / Gating Mechanism
       A component in neural networks (often using sigmoid activation)
       that controls the flow of information through a layer, allowing
       the network to dynamically adjust computations. See :term:`GLU`
       and :term:`GRN`.

   GLU (Gated Linear Unit)
       A specific gating mechanism, often :math:`a \odot \sigma(b)`, where
       :math:`a` and :math:`b` are linear transformations of an input,
       :math:`\odot` is element-wise multiplication, and :math:`\sigma`
       is sigmoid. Used within :term:`GRN`.

   GRN (Gated Residual Network)
       A core component combining linear transformations, non-linear
       activation, gating (GLU), and a residual connection with layer
       normalization. Enables complex, stable transformations. See
       :class:`~fusionlab.nn.components.GatedResidualNetwork`.

   Heuristic
       A practical rule or method, often based on experience, used when
       an optimal algorithm is impractical (e.g., heuristic choice of
       decomposition model).

   Hierarchical Attention
       An attention mechanism designed to process inputs at multiple
       levels or scales, potentially capturing relationships within and
       between different temporal resolutions or feature groups. See
       :class:`~fusionlab.nn.components.HierarchicalAttention`.

   Hyperparameter
       A parameter set *before* training begins, controlling model
       architecture or the learning process (e.g., learning rate,
       `hidden_units`). Contrast with model weights learned during training.

   Hyperparameter Tuning / Optimization
       The process of searching for the optimal set of hyperparameters
       to maximize model performance. See
       :doc:`/user_guide/forecast_tuner`.

   IQR (Interquartile Range)
       A measure of statistical dispersion (:math:`Q3 - Q1`). Used in
       robust outlier detection. See
       :func:`~fusionlab.utils.ts_utils.ts_outlier_detector`.

   Interpretability
       The degree to which a model's predictions and internal workings
       can be understood by humans. TFT/XTFT incorporate components like
       VSNs and attention to enhance interpretability.

   Keras
       A high-level API for building and training neural networks, commonly
       used with backends like TensorFlow, JAX, or PyTorch. `fusionlab`
       currently uses the Keras API provided by TensorFlow.

   Keras Tuner
       A library for automating hyperparameter tuning for Keras models.
       See :doc:`/user_guide/forecast_tuner`.

   KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin Test)
       A statistical test for :term:`Stationarity`. The null hypothesis
       is stationarity around a deterministic trend. A low p-value
       suggests non-stationarity. See
       :func:`~fusionlab.utils.ts_utils.trend_analysis`.

   Lag Features
       Features created by shifting a time series (:math:`Y_{t-k}`). See
       :func:`~fusionlab.utils.ts_utils.create_lag_features`.

   Latent Space / Representation
       A typically lower-dimensional space capturing salient features,
       learned by encoding high-dimensional data. Used in :term:`Autoencoder`.

   Layer Normalization
       Normalization applied across features for a *single* sample, often
       used in Transformers and GRNs. Contrast with :term:`Batch Normalization`.

   Learned Normalization
       Normalization using learned scale and shift parameters instead of
       pre-calculated statistics. See
       :class:`~fusionlab.nn.components.LearnedNormalization`.

   Lookback Period / Window
       The number of past time steps (:math:`T` or `time_steps`) used as
       input to predict the future.

   LOESS (Locally Estimated Scatterplot Smoothing)
       A non-parametric regression method fitting smooth curves locally.
       Used internally by :term:`STL`.

   Loss Function
       A function measuring the discrepancy between model predictions and
       true values, guiding model training via optimization. See
       :doc:`/user_guide/losses`.

   LSTM (Long Short-Term Memory)
       A type of Recurrent Neural Network (RNN) adept at learning long-range
       dependencies in sequences. See
       :class:`~fusionlab.nn.components.MultiScaleLSTM`.

   MAE (Mean Absolute Error)
       An evaluation metric: mean of absolute differences between
       predictions and actuals.

   Memory-Augmented Attention
       Attention mechanism incorporating an external, trainable memory
       matrix, allowing the model to potentially access longer-term or
       learned contextual information. See
       :class:`~fusionlab.nn.components.MemoryAugmentedAttention`.

   MinMaxScaler
       A scikit-learn scaler that transforms features to a specific range,
       typically [0, 1].

   MSE (Mean Squared Error)
       A common loss function/metric: mean of squared differences between
       predictions and actuals.

   Multi-Head Attention
       Attention performed multiple times in parallel using different
       projections (heads), allowing focus on different representation
       subspaces. See :ref:`Attention Mechanisms <user_guide_components>`.

   Multi-Horizon Forecasting
       Predicting multiple future time steps simultaneously. Requires
       `forecast_horizon` > 1.

   Multi-Modal Embedding
       A layer that projects multiple input sequences (modalities) into a
       common embedding space before combining them. See
       :class:`~fusionlab.nn.components.MultiModalEmbedding`.

   Multi-Resolution Attention Fusion
       A self-attention layer applied to features combined from various
       sources (e.g., multi-scale LSTMs, different attention outputs) to
       create a unified representation. See
       :class:`~fusionlab.nn.components.MultiResolutionAttentionFusion`.

   Multi-Scale Processing
       Analyzing data at different temporal resolutions simultaneously.
       See :class:`~fusionlab.nn.components.MultiScaleLSTM`.

   Multi-Target Forecasting
       Predicting multiple related target variables simultaneously.
       See :func:`~fusionlab.datasets.make.make_multivariate_target_data`.

   Multiplicative Decomposition
       Time series decomposition where components are multiplied:
       :math:`Y_t = T_t \times S_t \times R_t`. See
       :func:`~fusionlab.utils.ts_utils.decompose_ts`.

   Multivariate Time Series
       A time series consisting of observations on multiple variables
       over time.

   NumPy Style Docstrings
       A convention for formatting Python docstrings using specific
       sections (Parameters, Returns, etc.). Used by `fusionlab` and
       parsed by :mod:`sphinx.ext.napoleon`.

   NTemporalFusionTransformer
       A ``fusionlab-learn`` variant of TFT requiring static and dynamic inputs,
       currently focused on point forecasts. See
       :class:`~fusionlab.nn.NTemporalFusionTransformer`.

   One-Hot Encoding
       Converting categorical integer features into binary vectors where
       only the element corresponding to the category is 1.

   Outlier
       A data point significantly different from other observations. See
       :func:`~fusionlab.utils.ts_utils.ts_outlier_detector`.

   PACF (Partial Autocorrelation Function)
       Measures correlation between a series and its lag, after removing
       effects of intermediate lags. Helps identify AR order. See
       :func:`~fusionlab.utils.ts_utils.ts_corr_analysis`.

   Pinball Loss
       See :term:`Quantile Loss`.

   Point Forecast
       A single value prediction for each future time step. Contrast with
       :term:`Quantile Forecast`.

   Positional Encoding
       Technique to inject sequence order information into models like
       Transformers that don't inherently process order. See
       :class:`~fusionlab.nn.components.PositionalEncoding`.

   Probabilistic Forecasting
       Forecasting that provides uncertainty estimates, typically via
       quantiles or a full predictive distribution. See
       :term:`Quantile Forecast`.

   Quantile
       A point below which a specified percentage (quantile level) of
       data falls (e.g., 0.5 quantile = median).

   Quantile Distribution Modeling
       The final output component in XTFT that maps decoder features to
       specific quantile predictions (or a point forecast). See
       :class:`~fusionlab.nn.components.QuantileDistributionModeling`.

   Quantile Forecast
       Predicting specific quantiles (e.g., 0.1, 0.5, 0.9) of the target
       variable's future distribution to represent uncertainty.

   Quantile Loss (Pinball Loss)
       Loss function for training quantile forecasting models, penalizing
       errors asymmetrically based on the quantile level. See
       :func:`~fusionlab.nn.losses.combined_quantile_loss`.

   R² Score (Coefficient of Determination)
       Statistical measure (:math:`R^2`) of the proportion of variance in the
       dependent variable predictable from independent variables.

   Recurrent Neural Network (RNN)
       A class of neural networks designed for sequential data, containing
       feedback loops (e.g., :term:`LSTM`, GRU).

   Residual
       The component of a time series remaining after Trend and Seasonality
       have been removed, or the error between predictions and actuals.

   Rolling Statistics / Window
       Statistics (mean, std dev) calculated over a sliding window. See
       :func:`~fusionlab.utils.ts_utils.ts_engineering`.

   Scaler
       Tool (e.g., `StandardScaler`, `MinMaxScaler`) for feature
       :term:`Scaling`.

   Scaling
       Transforming numerical features to a common scale (e.g., [0, 1] or
       mean 0, std 1) for better model training.

   Scikit-learn
       A popular Python library for machine learning, providing tools for
       preprocessing, model selection, evaluation, and various algorithms.
       Used by some ``fusionlab-learn`` utilities.

   SDT (Seasonal Decomposition of Time series)
       Classical time series decomposition method (additive/multiplicative).
       See :func:`~fusionlab.utils.ts_utils.decompose_ts`.

   Seasonality
       Patterns repeating over a fixed period (daily, weekly, yearly).

   Self-Attention
       Attention mechanism where a sequence attends to itself to model
       internal relationships.

   Sequence Length
       See :term:`Lookback Period / Window`.

   Sequence-to-Sequence (Seq2Seq) Model
       Architecture mapping an input sequence to an output sequence.

   Spatiotemporal Data
       Data that has both spatial (location) and temporal (time) dimensions.

   StandardScaler
       Scikit-learn scaler standardizing features to zero mean and unit
       variance (Z-score).

   Static Features
       Time-invariant features associated with a series (e.g., sensor ID,
       location category). Used as context by TFT/XTFT.

   Stationarity
       Property where a time series' statistical properties (mean, variance,
       autocorrelation) are constant over time.

   Statsmodels
       A Python library providing classes and functions for estimating
       many different statistical models, as well as statistical tests
       and data exploration. Used by some ``fusionlab-learn`` utilities.

   STL (Seasonal-Trend decomposition using LOESS)
       Robust time series decomposition method. See
       :func:`~fusionlab.utils.ts_utils.decompose_ts`.

   Supervised Learning
       Machine learning where a model learns a mapping from inputs to
       outputs using labeled examples. Forecasting is often framed this way.

   SuperXTFT
       An experimental, enhanced version of XTFT with input VSNs and extra
       GRNs. See :class:`~fusionlab.nn.SuperXTFT`.

   Taylor Diagram
       A diagram used to graphically summarize how well patterns match
       each other in terms of correlation, standard deviation, and RMSE.
       *(Mentioned in relation to k-diagram)*

   TensorFlow
       Open-source machine learning framework used as the backend for
       ``fusionlab-learn`` neural network models.

   TFT (Temporal Fusion Transformer)
       Baseline interpretable deep learning architecture for multi-horizon
       time series forecasting. See
       :class:`~fusionlab.nn.TemporalFusionTransformer` and
       :class:`~fusionlab.nn.transformers.TFT`.

   Time Series
       A sequence of data points indexed in time order.

   Time Series Cross-Validation
       Cross-validation respecting temporal order, typically using
       expanding or rolling forecast origins. See
       :func:`~fusionlab.utils.ts_utils.ts_split` (`split_type='cv'`).

   Trend
       The long-term increase or decrease in a time series.

   Uncertainty Quantification (UQ)
       The process of estimating and characterizing the uncertainty
       associated with model predictions, often via prediction intervals
       or full distributions. Quantile forecasting is a method for UQ.

   Univariate Time Series
       A time series with observations on only a single variable.

   VSN (Variable Selection Network)
       Component that learns importance weights for input features. See
       :class:`~fusionlab.nn.components.VariableSelectionNetwork`.

   XTFT (Extreme Temporal Fusion Transformer)
       Enhanced TFT variant with multi-scale processing, advanced attention,
       and anomaly detection. See :class:`~fusionlab.nn.XTFT`.

   Z-Score
       Statistical measure of a value's deviation from the mean in units
       of standard deviations. See
       :func:`~fusionlab.utils.ts_utils.ts_outlier_detector`.