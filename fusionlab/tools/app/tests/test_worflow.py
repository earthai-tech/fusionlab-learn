
import pytest 
import os 
import shutil 
import tempfile, json
from pathlib import Path
from fusionlab.utils.deps_utils import get_versions 
from fusionlab.tools.app.config import SubsConfig 
from fusionlab.tools.app.processing import DataProcessor, SequenceGenerator 
from fusionlab.tools.app.modeling import ModelTrainer, Forecaster  
from fusionlab.tools.app.view import ResultsVisualizer 
from fusionlab.tools.app.inference import PredictionPipeline 
#%
def test_fwd_worflow ():
    print("--- Starting Subsidence Forecasting Workflow ---")
    
    
    # 1. Configure the workflow
    # For this test, point to the sample data that comes with the library
    config = SubsConfig(
        data_dir='data/', #'_pinn_works/test_data', #'../../fusionlab/datasets/data',
        data_filename='zhongshan_500k.csv',
        epochs=3, # Use a small number of epochs for a quick test run
        save_intermediate=True,
        verbose=1
    )
    print(f"Configuration loaded for model '{config.model_name}'")
    # 2. Process Data
    processor = DataProcessor(config=config)
    processed_df = processor.run()
    
    # 3. Generate Sequences and Datasets
    sequence_gen = SequenceGenerator(config=config)
    train_dataset, val_dataset = sequence_gen.run(
        processed_df, processor.static_features_encoded
    )
    # 4. Train the Model
    # Get input shapes from a sample batch for model instantiation
    sample_inputs, _ = next(iter(train_dataset))
    input_shapes = {name: tensor.shape for name, tensor in sample_inputs.items()}
    
    trainer = ModelTrainer(config=config)
    best_model = trainer.run(train_dataset, val_dataset, input_shapes)
    
    # 5. Make Forecasts
    forecaster = Forecaster(config=config)
    forecast_df = forecaster.run(
        model=best_model,
        test_df=sequence_gen.test_df,
        val_dataset=val_dataset,
        static_features_encoded=processor.static_features_encoded,
        coord_scaler=sequence_gen.coord_scaler
    )
    
    # 6. Visualize Results
    visualizer = ResultsVisualizer(config=config)
    visualizer.run(forecast_df)
    
    print("\n--- Workflow Finished Successfully ---")

def test_inference ():
    
    print("--- Starting Full Training & Prediction Workflow ---")
    # ==================================================================
    # Part 1: Full Training Workflow
    # ==================================================================
    
    # 1. Configure the workflow
    # For this test, we create a temporary directory for all outputs.
    # Note: Using a smaller dataset for faster testing is recommended.
    output_directory = "./app_workflow_test_run"
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory) # Clean up previous runs
    
    config = SubsConfig(
        # For a quick test, we can use the smaller built-in dataset
        # by pointing to a non-existent file and letting it fallback.
        data_dir='./data', 
        data_filename='zhongshan_500k.csv',
        epochs=3, # Use a small number of epochs for a quick test
        output_dir=output_directory,
        save_intermediate=True, # Ensure artifacts are saved
        verbose=1
    )
    print(f"Configuration loaded for model '{config.model_name}'")
    print(f"All artifacts will be saved in: {config.run_output_path}")
    
    # 2. Process Data
    # This will load data, preprocess it, and save the encoder and scaler.
    processor = DataProcessor(config=config)
    processed_df = processor.run()
    
    # 3. Generate Sequences and Datasets
    # This will create sequences and save the coordinate scaler.
    sequence_gen = SequenceGenerator(config=config)
    train_dataset, val_dataset = sequence_gen.run(
        processed_df, processor.static_features_encoded
    )
    
    # 4. Train the Model
    # This will train the model and save the best version.
    sample_inputs, _ = next(iter(train_dataset))
    input_shapes = {name: tensor.shape for name, tensor in sample_inputs.items()}
    
    trainer = ModelTrainer(config=config)
    best_model = trainer.run(train_dataset, val_dataset, input_shapes)
    
    print("\n--- Training Workflow Finished Successfully ---")
    
    # XXXXX THE INFERENCE WORK FLOW. 
    # ==================================================================
    # Part 2: Inference Workflow using PredictionPipeline
    # ==================================================================
    print("\n" + "="*50)
    print("--- Starting Inference Workflow on Validation Data ---")
    print("="*50 + "\n")
    
    # Define the paths to the artifacts we just saved in the training run
    model_path = os.path.join(
        config.run_output_path, f"{config.model_name}.keras")
    encoder_path = os.path.join(
        config.run_output_path, "ohe_encoder.joblib")
    scaler_path = os.path.join(
        config.run_output_path, "main_scaler.joblib")
    # Note: The SequenceGenerator saves the coord scaler in its own file
    coord_scaler_path = os.path.join(
        config.run_output_path, "coord_scaler.joblib")
    
    # We will use the original data file as the "new" data to predict on.
    # The loader inside the pipeline needs a valid source.
    validation_data_path = processor.raw_df_path 
    
    # Check if all required artifacts exist before proceeding
    required_artifacts = [model_path, encoder_path, scaler_path, coord_scaler_path]
    if not all(os.path.exists(p) for p in required_artifacts):
        print("[ERROR] Not all required artifacts from the training run were found.")
        print("Skipping prediction pipeline.")
    else:
        # Instantiate the prediction pipeline with the paths to the artifacts
        prediction_pipeline = PredictionPipeline(
            config=config,
            model_path=model_path,
            encoder_path=encoder_path,
            scaler_path=scaler_path,
            coord_scaler_path=coord_scaler_path
        )
        
        # Run the entire prediction and visualization workflow
        prediction_pipeline.run(validation_data_path=validation_data_path)
        

def test_inference2():
    print("\n=== 1) TRAINING WORKFLOW =========================================")

    
    import pytest 
    import os 
    import shutil 
    import tempfile, json
    from pathlib import Path
    from fusionlab.utils.deps_utils import get_versions 
    from fusionlab.tools.app.config import SubsConfig 
    from fusionlab.tools.app.processing import DataProcessor, SequenceGenerator 
    from fusionlab.tools.app.modeling import ModelTrainer, Forecaster  
    from fusionlab.tools.app.view import ResultsVisualizer 
    from fusionlab.tools.app.inference import PredictionPipeline 

    # 0. scratch directory – every test run is self-contained 
    work_dir = Path(tempfile.mkdtemp(prefix="flab_gui_test_"))
    print(f"[tmp]  all artefacts go to:  {work_dir}")

    #
    # 1. configuration (tiny run – just to exercise the pipeline) 
    cfg = SubsConfig(
        data_dir='./data', 
        data_filename='zhongshan_500k.csv',
        output_dir    = str(work_dir),
        epochs        = 3,
        batch_size    = 64,
        save_intermediate = True,
        verbose       = 0,
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
    manifest_path = (
        Path(cfg.registry_path) / "run_manifest.json"
        )
    assert manifest_path.exists(), "manisfer file – cannot test inference"
    
    pipe = PredictionPipeline(
        # manifest_path = manifest_path,             # <── all paths auto-filled
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