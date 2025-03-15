import contextlib
import os
import py_compile
import subprocess
import time
from pathlib import Path

import click
import joblib
from joblib import Parallel, delayed
from tqdm.auto import tqdm


# joblibで用いるtqdm用の関数を定義
@contextlib.contextmanager
def tqdm_joblib(total, **kwargs):
    progress_bar = tqdm(total=total, smoothing=0, **kwargs)

    class TqdmBatchCompletionCallBack(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            progress_bar.update(n=self.batch_size)

            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallBack

    try:
        yield progress_bar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        progress_bar.close()


def run_case(
    case_no,
    workspace_dir,
    main_script_path,
):
    input_path = workspace_dir / "io" / "in" / f"{case_no:04d}.txt"
    output_path = workspace_dir / "io" / "out" / f"{case_no:04d}.txt"
    error_path = workspace_dir / "io" / "err" / f"{case_no:04d}.txt"

    # Run the main.py script with the specified input and output redirection
    start_time = int(time.time() * 1000)
    with input_path.open("r") as infile, output_path.open("w") as outfile, error_path.open("w") as errfile:
        subprocess.run(["python3", main_script_path], stdin=infile, stdout=outfile, stderr=errfile)
    end_time = int(time.time() * 1000)
    time_ms = end_time - start_time

    # Extract the score from the error file
    with error_path.open("r") as errfile:
        score = None
        for line in errfile:
            if "score" in line:
                score = line.split()[-1]
                break

    return case_no, score, time_ms


@click.command()
@click.argument("start_case", default=0)
@click.argument("end_case", default=99)
def main(start_case, end_case):
    total_cases = end_case - start_case + 1
    workspace_dir = Path(os.getenv("WORKSPACE_DIR", "."))
    main_script_path = workspace_dir / "python" / "main.py"

    py_compile.compile(main_script_path)
    with tqdm_joblib(total=total_cases, desc="Cases"):
        results = Parallel(n_jobs=-1)(
            delayed(run_case)(
                case_no,
                workspace_dir,
                main_script_path,
            )
            for case_no in range(start_case, end_case + 1)
        )

    total_score = 0
    total_time = 0
    max_time = 0
    min_time = float("inf")
    max_time_case = None
    error_count = 0
    error_cases = []

    for case_no, score, time_ms in results:
        if score is None:
            error_count += 1
            error_cases.append(case_no)
        else:
            total_score += int(score)
        total_time += time_ms
        if time_ms > max_time:
            max_time = time_ms
            max_time_case = case_no
        if time_ms < min_time:
            min_time = time_ms

    N = end_case - start_case + 1
    mean_time = total_time / N

    print("###########################################")
    print(f" Total score \t:\t {total_score}")
    print(" Execution time (ms):")
    print(f"  Max \t:\t {max_time} ms (Case: {max_time_case})")
    print(f"  Min \t:\t {min_time} ms")
    print(f"  Mean \t:\t {mean_time:.2f} ms")
    if error_count > 0:
        print("\n Error information:")
        print(f"  Error count \t:\t {error_count}")
        print(f"  Error cases \t:\t {error_cases}")
    print("###########################################")


if __name__ == "__main__":
    main()
