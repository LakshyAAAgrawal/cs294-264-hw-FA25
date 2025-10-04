#!/usr/bin/env python3
import concurrent.futures
import subprocess
from pathlib import Path

import typer
from datasets import load_dataset

from utils import save_traj, update_preds_file, remove_from_preds_file, get_sb_environment

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DATASET_MAPPING = {
    "cs294": "lynnliu030/swebench-eval-subset",
}

from agent import ReactAgent
from llm import OpenAIModel
from response_parser import ResponseParser
from envs import SWEEnvironment, DumbEnvironment

def process_instance(
    instance: dict,
    output_dir: Path,
    model_name: str,
    max_steps: int,
) -> None:
    """Process a single SWEBench instance."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    
    # Avoid inconsistent state if something here fails and there's leftover previous files
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)
    
    # Initialize the model and parser
    llm = OpenAIModel(ResponseParser.END_CALL, model_name)
    parser = ResponseParser()
    task = instance["problem_statement"]
    
    print(f"Processing instance {instance_id}")
    agent = None    
    result = ""
    
    try:
        # Initialize the environment
        env = SWEEnvironment(instance)
        try:
            output = env.run_bash_cmd("pip install -q pyflakes 2>/dev/null || true")  # Quiet install, don't fail if it errors
            print(f"Installed pyflakes: {output}")
        except Exception:
            import traceback
            traceback.print_exc()
            pass
        # Initialize the agent with verification and LLM judge functions
        agent = ReactAgent(
            "swe-agent", 
            parser, 
            llm, 
            instance_id=instance_id, 
            output_dir=output_dir, 
            verify_code_quality_fn=env.verify_before_finish,
            llm_as_judge_fn=env.llm_judge_validate_changes
        )
        
        agent.add_functions([
            env.run_bash_cmd, 
            env.show_file,
            env.search_in_file,
            env.search_in_directory,
            env.list_files,
            env.find_file,
            env.replace_in_file,
            # env.set_file_content,           # ← For complete file rewrites
            env.replace_between,            # ← For pattern-based editing (more robust)
            env.insert_lines_at,
            env.delete_lines,
            env.git_diff,
            env.generate_patch,
            env.run_tests,
            env.check_repo_syntax,
            env.git_apply,
            env.run_python_snippet,
            env.detect_indentation,
            env.list_modified_python_files,
            env.find_and_replace_text,
            env.count_lines,                # ← Check file size before reading
            env.verify_before_finish        # ← Comprehensive pre-finish checks
        ])

        # Run the agent
        output = agent.run(task, max_steps) 
        
        # Generate patch for SWE-Bench
        result = env.generate_patch(output)
        
    except Exception as e:
        print(f"Error processing instance {instance_id}: {e}")
        
    finally:
        # Save the trajectory and update the predictions file
        save_traj(
            agent,
            instance_dir / f"{instance_id}.traj.json",
            result=result,
            instance_id=instance_id,
        )
        update_preds_file(output_dir / "preds.json", instance_id, model_name, result)
        print(f"Completed instance {instance_id}, result: {result}")

@app.command(help="Run CS294 HW on subset of SWEBench instances.")
def main(
    subset: str = typer.Option("cs294", "--subset", help="SWEBench subset used or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("test", "--split", help="Dataset split", rich_help_panel="Data selection"),
    output: str = typer.Option("outputs", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    model_name: str = typer.Option("gpt-5-mini", "--model", help="Model used", rich_help_panel="Basic"),
    max_steps: int = typer.Option(100, "--max-steps", help="Maximum number of steps", rich_help_panel="Basic"),
    # NOTE: provide any extra arguments if needed
) -> None:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to {output_path}")

    dataset_path = DATASET_MAPPING.get(subset, subset)
    print(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))
    print(f"Running on {len(instances)} instances...")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                print(f"Error in future for instance {instance_id}: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(process_instance, instance, output_path, model_name, max_steps): instance[
                "instance_id"
            ]
            for instance in instances
        }
        try:
            process_futures(futures)
        except KeyboardInterrupt:
            print("Cancelling all pending jobs. Press ^C again to exit immediately.")
            for future in futures:
                if not future.running() and not future.done():
                    future.cancel()
            process_futures(futures)


if __name__ == "__main__":
    app()