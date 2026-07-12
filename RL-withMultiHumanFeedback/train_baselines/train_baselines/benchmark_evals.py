import argparse
import os
import traceback
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml

yaml.add_multi_constructor("!!", lambda loader, suffix, node: None)
yaml.add_multi_constructor(
    "tag:yaml.org,2002:python/name",
    lambda loader, suffix, node: None,
    Loader=yaml.SafeLoader,
)


def collect_results(
    model_base_path: str = "gt_agents",  # Updated default path
    algorithms: List[str] = ["ppo", "sac"],
    output_dir: Optional[str] = None,
    full_param_df: bool = False,
    seed: Optional[int] = None,
    env: Optional[str] = None,
) -> pd.DataFrame:
    """
    Collect and process results from multiple algorithm runs.

    Args:
        model_base_path: Base path for model results
        algorithms: List of algorithms to process
        output_dir: Directory to save results (if None, uses model_base_path)

    Returns:
        DataFrame containing processed results
    """
    if not os.path.isabs(model_base_path):
        model_base_path = os.path.abspath(model_base_path)

    # Process each algorithm
    all_results = pd.DataFrame()
    if len(algorithms) > 0:
        for algorithm in algorithms:
            print(f"Processing algorithm: {algorithm}")
            algorithm_df = process_single_run_results(
                model_base_path, algorithm, full_param_df=full_param_df, seed=seed, env=env
            )
            all_results = pd.concat([all_results, algorithm_df], ignore_index=True)
    else:
        algorithm_df = process_single_run_results(
            model_base_path, None, full_param_df=full_param_df, seed=seed, env=env
        )
        all_results = pd.concat([all_results, algorithm_df], ignore_index=True)

    # Sort and save results
    if not all_results.empty:
        all_results = all_results.sort_values(["environment", "algorithm"])
        if output_dir is None:
            output_dir = model_base_path
        else:
            output_dir = os.path.abspath(output_dir)
        output_path = os.path.join(output_dir, "collected_results.csv")
        all_results.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    else:
        print("No results were collected!")

    return all_results

# --- Updated function to process results for a specific algorithm ---
def safe_yaml_load(path: str):
    """Loads a YAML file, skipping the first line."""
    with open(path) as theyaml:
        next(theyaml)
        return yaml.safe_load(theyaml)
    
# --- Updated function to process single run results directly from run folder ---
def process_single_run_results(
    # 🎯 model_base_path should now point directly to the run folder, 
    # e.g., "gt_agents/sac/Hopper-v5_42"
    model_base_path: str, 
    algorithm: Optional[str] = None, 
    full_param_df: bool = False,
    env: Optional[str] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Processes results for a single run by taking the run directory as the 
    base path. It reads the seed value from the args.yml file within.
    """
    df = pd.DataFrame()
    full_run_dir = f"{model_base_path}/{algorithm}/{env}_{seed}" # The entire run path is now the input

    if not os.path.exists(full_run_dir):
        print(f"Path not found for run: {full_run_dir}")
        return df

    # --- Extract environment name and seed parts from directory name ---
    # This is necessary for file path construction/metadata, even if we skip listdir.
    run_dir_name = os.path.basename(full_run_dir)
    parts = run_dir_name.rsplit('_', 1)
    env_name_from_dir = parts[0] if len(parts) > 1 and parts[-1].isdigit() else run_dir_name
    
    try:
        print(f"Looking for args at: {full_run_dir}")

        # 1. Check for configuration files
        args_path = os.path.join(full_run_dir, "args.yml")
        print(f"Looking for args_path at: {args_path}")

        config_path = os.path.join(full_run_dir, "config.yml")
        print(f"Looking for config at: {config_path}")

        
        # Fallback for old/nested structure
        if not os.path.isfile(args_path):
            old_args_path = os.path.join(full_run_dir, env_name_from_dir, "args.yml")
            if os.path.isfile(old_args_path):
                args_path = old_args_path
                config_path = os.path.join(full_run_dir, env_name_from_dir, "config.yml")
            else:
                print(f"Args file not found in expected locations for: {full_run_dir}")
                return df
        
        # 2. Read YAML files
        run_arguments = safe_yaml_load(args_path)
        config = safe_yaml_load(config_path)

        # Combine all parameters
        the_dict = {}
        for elem in run_arguments[0]:
            the_dict[elem[0]] = elem[1]
        for elem in config[0]:
            the_dict[elem[0]] = elem[1]

        # 3. Load evaluation results
        eval_path = os.path.join(full_run_dir, "evaluations.npz")
        if not os.path.isfile(eval_path):
            print(f"Evaluation file not found: {eval_path}")
            evals = np.nan
        else:
            evals = np.mean(np.load(eval_path)["results"][-1])

        # 4. Create result dictionary (SEED IS DERIVED FROM THE_DICT)
        seed_value = the_dict.get("seed", int(parts[-1]) if parts[-1].isdigit() else 0)
        
        # Use the algorithm parameter if provided, otherwise derive from the config
        alg_name = algorithm if algorithm else the_dict.get("algorithm", the_dict.get("algo", "N/A"))
        
        result_dict = {
            "algorithm": alg_name,
            "environment": the_dict.get("environment", the_dict.get("env", env_name_from_dir)),
            "run": run_dir_name,
            "seed": seed_value,
            "eval_score": evals,
        }
        
        if full_param_df:
             result_dict = {**the_dict, **result_dict}

        # 5. Append to dataframe and return
        return pd.concat([df, pd.DataFrame([result_dict])], ignore_index=True)

    except Exception as e:
        print(f"Error processing {full_run_dir}:")
        print(traceback.format_exc())
        return df
    
    
def main():
    parser = argparse.ArgumentParser(
        prog="CollectResults",
        description="Collecting results from multiple algorithm runs",
    )
    parser.add_argument(
        "--model-base-path", default="train_baselines/gt_agents"
    )  # Updated default path
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["ppo", "sac"],
        help="List of algorithms to process (default: ppo sac)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save results (default: same as model-base-path)",
    )

    args = parser.parse_args()
    collect_results(args.model_base_path, args.algorithms, args.output_dir)


if __name__ == "__main__":
    main()
