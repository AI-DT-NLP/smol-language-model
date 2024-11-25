import subprocess
import os

def run_pipeline(pipeline_name):
    """
    Run the specified pipeline: preprocessing -> model -> train.

    Args:
        pipeline_name (str): Name of the pipeline to execute.
    """
    print(f"=== Starting Pipeline: {pipeline_name} ===")

    # Step 1: Preprocessing
    preprocess_script = f"preprocessing/{pipeline_name}.py"
    if os.path.exists(preprocess_script):
        print(f"Running Preprocessing: {preprocess_script}")
        subprocess.run(["python", preprocess_script], check=True)
    else:
        print(f"Preprocessing script not found: {preprocess_script}")
        return

    # Step 2: Model Initialization
    model_script = f"model/{pipeline_name}.py"
    if os.path.exists(model_script):
        print(f"Running Model Initialization: {model_script}")
        subprocess.run(["python", model_script], check=True)
    else:
        print(f"Model script not found: {model_script}")
        return

    # Step 3: Training
    train_script = f"train/{pipeline_name}.py"
    if os.path.exists(train_script):
        print(f"Running Training: {train_script}")
        subprocess.run(["python", train_script], check=True)
    else:
        print(f"Training script not found: {train_script}")
        return

    print(f"=== Pipeline {pipeline_name} Completed Successfully ===")


if __name__ == "__main__":
    # Specify the pipeline name
    pipeline_name = "baseline"  # Change this to run a different pipeline
    run_pipeline(pipeline_name)
