Daniel@DESKTOP-J0II3A7 MINGW64 /f/repositories/fusionlab-learn (develop)
$ python -m scripts.plot_physics_sanity --src "J:\nature\results\nansha_GeoPriorSubsNet_stage1\train_20260222-141331" --src "J:\nature\results\zhongshan_GeoPriorSubsNet_stage1\train_20260218-175001" --out fig4_physics_sanity --out-json fig4_physics_sanity.json --extra k-from-tau,closure --city Nansha --city Zhongshan


Daniel@DESKTOP-J0II3A7 MINGW64 /f/repositories/fusionlab-learn (develop)
$ python -m scripts.plot_physics_sanity --src "J:\nature\results\nansha_GeoPriorSubsNet_stage1\train_20260222-141331" --src "J:\nature\results\zhongshan_GeoPriorSubsNet_stage1\train_20260218-175001" --out fig4_physics_sanity_scaled --out-json fig4_physics_sanity_scaled.json --extra k-from-tau,closure --city Nansha --city Zhongshan --cons-kind scaled

Daniel@DESKTOP-J0II3A7 MINGW64 /f/repositories/fusionlab-learn (develop)
$ python -m scripts.plot_physics_sanity --src "J:\nature\results\nansha_GeoPriorSubsNet_stage1\train_20260222-141331" --src "J:\nature\results\zhongshan_GeoPriorSubsNet_stage1\train_20260218-175001" --out fig4_physics_sanity_residual_scaled --out-json fig4_physics_sanity_residual_scaled.json --extra k-from-tau,closure --city Nansha --city Zhongshan --plot-mode residual --cons-kind scaled


$ python -m scripts.plot_physics_fields --payload "J:\nature\results\zhongshan_GeoPriorSubsNet_stage1\train_20260218-175001\zhongshan_phys_payload_run_val.npz" --coords-npz "J:\nature\results\zhongshan_GeoPriorSubsNet_stage1\artifacts\val_inputs.npz" --out fig6_physics_fields_zhongshan-none --out-json fig6_physics_field_zhongshan-none.json --render hexbin --show-ticklabels false --show-labels false

$ python -m scripts.plot_physics_fields --payload "J:\nature\results\nansha_GeoPriorSubsNet_stage1\train_20260222-141331\nansha_phys_payload_run_val.npz" --coords-npz "J:\nature\results\nansha_GeoPriorSubsNet_stage1\artifacts\val_inputs.npz" --out fig6_physics_fields_nansha-none --out-json fig6_physics_field_nansha-none.json --render hexbin --show-ticklabels false --show-labels false

$ python -m scripts.plot_core_ablation --ns-with "J:\nature\results\nansha_GeoPriorSubsNet_stage1\train_20260222-141331" --ns-no "J:\nature\results\nansha_GeoPriorSubsNet_stage1\train_20260223-135903" --zh-with "J:\nature\results\zhongshan_GeoPriorSubsNet_stage1\train_20260218-175001" --zh-no "J:\nature\results\zhongshan_GeoPriorSubsNet_stage1\train_20260221-103340" --core-metric mae --cities Nansha,zhongshan  --show-legend false --show-labels false --show-values false

$ python -m scripts.plot_uncertainty --ns-forecast "J:\nature\results\nansha_GeoPriorSubsNet_stage1\train_20260222-141331\nansha_GeoPriorSubsNet_forecast_TestSet_H3_eval_calibrated.csv" --zh-forecast "J:\nature\results\zhongshan_GeoPriorSubsNet_stage1\train_20260218-175001\zhongshan_GeoPriorSubsNet_forecast_TestSet_H3_eval_calibrated.csv" --split auto --ns-phys-json  "J:\nature\results\nansha_GeoPriorSubsNet_stage1\train_20260222-141331\nansha_GeoPriorSubsNet_eval_diagnostics_TestSet_H3_calibrated.json" --zh-phys-json "J:\nature\results\zhongshan_GeoPriorSubsNet_stage1\train_20260218-175001\zhongshan_GeoPriorSubsNet_eval_diagnostics_TestSet_H3_calibrated.json" --show-labels false --show-point-values false --show-mini-legend false

$ python -m scripts.plot_uncertainty_extras --ns-forecast "J:\nature\results\nansha_GeoPriorSubsNet_stage1\train_20260222-141331\nansha_GeoPriorSubsNet_forecast_TestSet_H3_eval_calibrated.csv" --zh-forecast "J:\nature\results\zhongshan_GeoPriorSubsNet_stage1\train_20260218-175001\zhongshan_GeoPriorSubsNet_forecast_TestSet_H3_eval_calibrated.csv" --split auto --ns-phys-json  "J:\nature\results\nansha_GeoPriorSubsNet_stage1\train_20260222-141331\geoprior_eval_phys_20260222-215049_interpretable.json" --zh-phys-json "J:\nature\results\zhongshan_GeoPriorSubsNet_stage1\train_20260218-175001\geoprior_eval_phys_20260220-172641_interpretable.json" --show-labels false

$ python -m scripts plot-physics-sensitivity --results-root "J:\nature\results" --render tricontour --trend-arrow true --trend-arrow-len 0.22 --trend-arrow-pos 0.78,0.14 --cities ns,zh --clip 2,98

$ python -m scripts build-model-metrics --results-root "J:\nature\results" --out model_metrics_all

$ python -m scripts plot-xfer-impact --src "F:\repositories\fusionlab-learn\results\xfer\nansha__zhongshan" --split val --calib source

$ python -m scripts plot-xfer-impact --src "F:\repositories\fusionlab-learn\results\xfer\nansha__zhongshan" --split val 

$ python -m scripts.fix_xfer_interval_metrics --src results/xfer

$ python -m scripts.plot_geo_cumulative --ns-val "J:\nature\results\nansha_GeoPriorSubsNet_stage1\train_20260222-141331\nansha_GeoPriorSubsNet_forecast_TestSet_H3_eval_calibrated.csv" --zh-val "J:\nature\results\zhongshan_GeoPriorSubsNet_stage1\train_20260218-175001\zhongshan_GeoPriorSubsNet_forecast_TestSet_H3_eval_calibrated.csv" --ns-future "J:\nature\results\nansha_GeoPriorSubsNet_stage1\train_20260222-141331\nansha_GeoPriorSubsNet_forecast_TestSet_H3_future_calibrated.csv" --zh-future "J:\nature\results\zhongshan_GeoPriorSubsNet_stage1\train_20260218-175001\zhongshan_GeoPriorSubsNet_forecast_TestSet_H3_future_calibrated.csv"

python -m scripts extend-forecast \
  --ns-src "J:\nature\results\nansha_GeoPriorSubsNet_stage1\train_20260222-141331" \
  --zh-src "J:\nature\results\zhongshan_GeoPriorSubsNet_stage1\train_20260218-175001" \
  --split test \
  --add-years 1 \
  --out future_plus1
  
$ python -m scripts summarize-hotspots   --hotspot-csv "F:\repositories\fusionlab-learn\scripts\out\fig6_hotspot_points.csv"

$ python -m scripts plot-spatial-forecasts   --ns-src "J:\nature\results\nansha_GeoPriorSubsNet_stage1\train_20260222-141331"   --zh-src "J:\nature\results\zhongshan_GeoPriorSubsNet_stage1\train_20260218-175001"   --split test   --year-val 2022   --years-forecast 2025 2026   --cumulative   --hotspot delta   --hotspot-quantile 0.90   --hotspot-out "scripts/out/fig6_hotspot_points.csv"   --out fig6-spatial-forecasts

python -m scripts plot-hotspot-analytics \
  --ns-src "J:\nature\results\nansha_GeoPriorSubsNet_stage1\train_20260222-141331" \
  --zh-src "J:\nature\results\zhongshan_GeoPriorSubsNet_stage1\train_20260218-175001" \
  --split test \
  --subsidence-kind cumulative \
  --base-year 2022 \
  --years 2023 2024 2025 \
  --risk-threshold 50 \
  --hotspot-quantile 0.90 \
  --out fig7-hotspot-analytics 

python -m scripts plot-hotspot-analytics \
  --ns-src "J:\nature\results\nansha_GeoPriorSubsNet_stage1\train_20260222-141331" \
  --zh-src "J:\nature\results\zhongshan_GeoPriorSubsNet_stage1\train_20260218-175001" \
  --split test \
  --subsidence-kind cumulative \
  --base-year 2022 \
  --years 2023 2024 2025 \
  --focus-year 2025 \
  --risk-threshold 50 \
  --hotspot-rule combo \
  --hotspot-abs 20 \
  --risk-min 0.7 \
  --exposure "scripts/out/exposure.csv" \
  --cluster-rank risk \
  --add-persistence \
  --boundary "scripts/out/nansha_boundary.geojson" \
  --timeline-mode ever \
  --timeline-overlay-current true \
  --out fig7-hotspot-analytics_ever_overlay
  

python -m scripts plot-hotspot-analytics \
  --ns-src "J:\nature\results\nansha_GeoPriorSubsNet_stage1\train_20260222-141331" \
  --zh-src "J:\nature\results\zhongshan_GeoPriorSubsNet_stage1\train_20260218-175001" \
  --split test \
  --subsidence-kind cumulative \
  --base-year 2022 \
  --years 2024 2025 \
  --focus-year 2025 \
  --risk-threshold 50 \
  --hotspot-rule combo \
  --hotspot-abs 20 \
  --risk-min 0.7 \
  --cluster-rank risk \
  --out fig7-hotspot-analytics_policy
  
python -m scripts make-boundary \
  --ns-src "J:\nature\results\nansha_GeoPriorSubsNet_stage1\train_20260222-141331" \
  --zh-src "J:\nature\results\zhongshan_GeoPriorSubsNet_stage1\train_20260218-175001" \
  --split test \
  --method concave \
  --alpha 0.25 \
  --format geojson \
  --out boundary 
  
python -m scripts make-exposure \
  --ns-src "J:\nature\results\nansha_GeoPriorSubsNet_stage1\train_20260222-141331" \
  --zh-src "J:\nature\results\zhongshan_GeoPriorSubsNet_stage1\train_20260218-175001" \
  --split test \
  --mode density \
  --k 30 \
  --out exposure 
  
python -m scripts make-district-grid \
  --ns-src "J:\nature\results\nansha_GeoPriorSubsNet_stage1\train_20260222-141331" \
  --split test \
  --nx 12 --ny 12 \
  --boundary "scripts/out/boundary_nansha.geojson" \
  --clip-boundary \
  --format geojson \
  --assign-samples \
  --out district_grid_clipped 
  
  
python -m scripts tag-clusters-with-zones \
  --clusters "scripts/out/hotspot_clusters.csv" \
  --grid "scripts/out/district_grid_nansha.geojson" \
  --out cluster_zones_nansha
  
# If your grid is clipped and some centroids fall outside, use nearest:

python -m scripts tag-clusters-with-zones \
  --clusters "scripts/out/hotspot_clusters.csv" \
  --grid "scripts/out/district_grid_nansha.geojson" \
  --nearest \
  --out cluster_zones_nansha
  
# If the clusters file contains both cities, filter:

python -m scripts tag-clusters-with-zones \
  --clusters "scripts/out/hotspot_clusters.csv" \
  --grid "scripts/out/district_grid_zhongshan.geojson" \
  --city "Zhongshan" \
  --out cluster_zones_zhongshan