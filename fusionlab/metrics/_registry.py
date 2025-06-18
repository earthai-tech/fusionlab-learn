
import importlib
from typing import Callable, Dict

# maps public name → fully-qualified path
_METRIC_PATHS: Dict[str, str] = {
    "coverage_score":                        "fusionlab.metrics.coverage_score",
    "weighted_interval_score":               "fusionlab.metrics.weighted_interval_score",
    "prediction_stability_score":            "fusionlab.metrics.prediction_stability_score",
    "time_weighted_mean_absolute_error":     "fusionlab.metrics.time_weighted_mean_absolute_error",
    "quantile_calibration_error":            "fusionlab.metrics.quantile_calibration_error",
    "mean_interval_width_score":             "fusionlab.metrics.mean_interval_width_score",
    "theils_u_score":                        "fusionlab.metrics.theils_u_score",
    "time_weighted_accuracy_score":          "fusionlab.metrics.time_weighted_accuracy_score",
    "time_weighted_interval_score":          "fusionlab.metrics.time_weighted_interval_score",
    "continuous_ranked_probability_score":   "fusionlab.metrics.continuous_ranked_probability_score",
}

def get_metric(name: str) -> Callable:
    """
    Return the metric function for `name`, lazily importing its module.
    
    Raises:
        ValueError: if `name` isn’t in the registry, with a list
                    of all valid metric names.
    """
    try:
        path = _METRIC_PATHS[name]
    except KeyError:
        valid = ", ".join(sorted(_METRIC_PATHS.keys()))
        raise ValueError(
            f"Unknown metric '{name}'. "
            f"Valid names are: {valid}"
        )
    module_name, func_name = path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, func_name)