
import pytest 
import shutil 
import tempfile, json
from pathlib import Path
# from fusionlab.registry  import _locate_manifest 
# from fusionlab.utils.deps_utils import get_versions 
# from fusionlab.tools.app.config import SubsConfig 
# from fusionlab.tools.app.processing import DataProcessor, SequenceGenerator 
# from fusionlab.tools.app.modeling import ModelTrainer #, Forecaster  
# # from fusionlab.tools.app.view import ResultsVisualizer 
# from fusionlab.tools.app.inference import PredictionPipeline 

def test_app_workflow ():

    
    from fusionlab.registry  import _locate_manifest 
    from fusionlab.utils.deps_utils import get_versions 
    from fusionlab.tools.app.config import SubsConfig 
    from fusionlab.tools.app.processing import DataProcessor, SequenceGenerator 
    from fusionlab.tools.app.modeling import ModelTrainer #, Forecaster  
    # from fusionlab.tools.app.view import ResultsVisualizer 
    from fusionlab.tools.app.inference import PredictionPipeline 
    
    print("\n=== 1) TRAINING WORKFLOW =========================================")
    # 0. scratch directory – every test run is self-contained 
    work_dir = Path(tempfile.mkdtemp(prefix="flab_gui_test_"))
    print(f"[tmp]  all artefacts go to:  {work_dir}")
    
    #
    # 1. configuration (tiny run – just to exercise the pipeline) 
    cfg = SubsConfig(
        data_dir='./data', 
        data_filename='nansha_500k.csv',
        output_dir    = str(work_dir),
        epochs        = 3,
        batch_size    = 64,
        save_intermediate = True,
        verbose       = 0,
        
        time_steps = 4, 
        train_end_year = 2021, 
        forecast_start_year = 2022,
        forecast_horizon_years = 3,
    
    )
    # No need to create will do it automatically 
    # manifest_path = (
    # Path(cfg.run_output_path) / "run_manifest.json"
    # )
    # ①  full static dump of the SubsConfig
    cfg.to_json(
        extra={"git": get_versions()},        # optional extras
    )
    
    # 2. end-to-end training
    processor  = DataProcessor(cfg) 
    df  = processor.run()
    seq_gen    = SequenceGenerator(cfg)
    tr_ds, va_ds = seq_gen.run(df, processor.static_features_encoded)
    sample_inp, _ = next(iter(tr_ds))
    shapes = {k: v.shape for k, v in sample_inp.items()}
    
    trainer    = ModelTrainer(cfg)
    _          = trainer.run(tr_ds, va_ds, shapes)
    
    print("✔ training finished")
    
    # ----------------------------------------------------------------------
    # 3. verify that the *manifest* was written ----------------------------
    # manifest_path = Path(cfg.run_output_path) / "run_manifest.json"
    # assert manifest_path.exists(), "manifest missing after training !"
    
    print("\n=== 2) INFERENCE WORKFLOW =======================================")
    
    # ----------------------------------------------------------------------
    # 4. use only the manifest + a 'new' CSV for prediction ----------------
    # here we just reuse the same raw CSV saved by the processor
    new_csv = Path(cfg.run_output_path) / "01_raw_data.csv"
    
    assert new_csv.exists(), "no raw CSV found – cannot test inference"
    
    manifest_path = _locate_manifest () 
    assert manifest_path.exists(), "manisfer file – cannot test inference"
    
    pipe = PredictionPipeline(
        manifest_path = manifest_path,             # <── all paths auto-filled
        kind ="inference", 
        log_callback  = print,
    )
    pipe.run(str(new_csv))
    
    print("\n✔ inference completed OK")
    
    # (optional) inspect what was appended to the manifest
    with open(manifest_path, "r", encoding="utf-8") as fp:
        print("\n--- manifest tail ---")
        print(json.dumps(json.load(fp)["inference"], indent=2))
    
    # ----------------------------------------------------------------------
    # 5. clean-up  (comment out if you want to keep the artefacts) ----------
    shutil.rmtree(work_dir, ignore_errors=True)

if __name__== "__main__": 
    pytest.main( [__file__,  "--maxfail=1 ", "--disable-warnings",  "-q"])