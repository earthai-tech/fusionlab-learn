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
