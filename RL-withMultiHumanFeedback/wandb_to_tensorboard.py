import os
import wandb
import tensorflow as tf
from tqdm import tqdm

# --- Configuration ---
# IMPORTANT: Provide a list of (entity, project_name) tuples you want to convert.
# Example: PROJECT_CONFIGS = [("rohit13595", "rl-project-hopper"), ("my_team", "rl-project-swimmer")]
PROJECT_CONFIGS = [
    ("YOUR_WANDB_ENTITY", "YOUR_WANDB_PROJECT_NAME") 
]
# Output directory where TensorBoard logs will be saved
TENSORBOARD_LOG_DIR = "./tensorboard_logs"
# ---------------------

def convert_wandb_to_tensorboard(project_configs: list, log_dir: str):
    """
    Fetches runs from multiple WandB projects (specified in project_configs) 
    and converts their metric history into local TensorBoard event files, 
    organizing them into subdirectories per project.
    """
    
    if not project_configs:
        print("ERROR: PROJECT_CONFIGS list is empty. Please define at least one (entity, project_name) tuple.")
        return

    # Check the placeholder configuration
    first_entity, first_project = project_configs[0]
    if first_entity == "YOUR_WANDB_ENTITY" or first_project == "YOUR_WANDB_PROJECT_NAME":
        print("ERROR: Please update the placeholder ENTITY and PROJECT_NAME in the PROJECT_CONFIGS list.")
        return

    api = wandb.Api()
    total_runs_found = 0
    total_runs_converted = 0
    
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Starting conversion process for {len(project_configs)} project(s)...")

    for entity, project_name in project_configs:
        print(f"\n--- Connecting to: {entity}/{project_name} ---")
        
        try:
            # Fetch all runs from the current project
            runs = api.runs(f"{entity}/{project_name}")
        except Exception as e:
            print(f"ERROR: Could not fetch runs for {entity}/{project_name}. Check entity/project name and API key. Details: {e}")
            continue

        if not runs:
            print(f"No runs found in project {project_name}. Skipping.")
            continue

        total_runs_found += len(runs)
        print(f"Found {len(runs)} runs. Starting conversion...")

        # Create a unique project-specific sub-directory inside the main log directory
        project_log_path = os.path.join(log_dir, f"{entity}_{project_name}")
        os.makedirs(project_log_path, exist_ok=True)
        
        converted_count = 0

        for run in tqdm(runs, desc=f"Converting {project_name}"):
            run_name = run.name
            run_id = run.id

            # Create a unique directory for this run's TensorBoard logs inside the project folder
            run_log_path = os.path.join(project_log_path, f"{run_name}-{run_id}")
            os.makedirs(run_log_path, exist_ok=True)

            try:
                # Create a SummaryWriter for this run
                writer = tf.summary.create_file_writer(run_log_path)
                
                # Fetch the entire history (all logged metrics)
                history = run.history()

                if history.empty:
                    tqdm.write(f"Skipping run {run_name} (ID: {run_id}): No metric history found.")
                    continue

                # Identify columns that are metrics (not system metadata like '_timestamp' or '_runtime')
                metric_cols = [col for col in history.columns if not col.startswith('_')]
                
                with writer.as_default():
                    for index, row in history.iterrows():
                        # The step index for TensorBoard is typically stored in the '_step' column in WandB
                        # If '_step' is not available, use the row index as a fallback (less accurate)
                        step = int(row.get('_step', index)) 

                        for metric_name in metric_cols:
                            metric_value = row[metric_name]
                            
                            # Only write scalar values (numbers)
                            if isinstance(metric_value, (int, float)):
                                tf.summary.scalar(metric_name, metric_value, step=step)
                
                writer.close()
                converted_count += 1
                total_runs_converted += 1

            except Exception as e:
                tqdm.write(f"ERROR processing run {run_name} (ID: {run_id}) in project {project_name}: {e}")
                continue
        
        print(f"Project {project_name} finished. Converted {converted_count} runs.")


    print(f"\nConversion complete.")
    print(f"Total Runs Found: {total_runs_found}")
    print(f"Total Runs Converted: {total_runs_converted}")
    print(f"TensorBoard logs saved to: {os.path.abspath(log_dir)}")
    print("To view all converted projects, run the following command in your terminal:")
    print(f"tensorboard --logdir {os.path.abspath(log_dir)}")

if __name__ == "__main__":
    convert_wandb_to_tensorboard(PROJECT_CONFIGS, TENSORBOARD_LOG_DIR)