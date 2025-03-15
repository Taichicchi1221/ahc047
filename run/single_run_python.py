import os
import py_compile
import subprocess
import time
from pathlib import Path

import click


@click.command()
@click.argument("case_no", default=0)
def main(case_no):
    # Define the paths
    workspace_dir = Path(os.getenv("WORKSPACE_DIR", "."))
    input_path = workspace_dir / "io" / "in" / f"{case_no:04d}.txt"
    output_path = workspace_dir / "io" / "out" / f"{case_no:04d}.txt"
    error_path = workspace_dir / "io" / "err" / f"{case_no:04d}.txt"
    main_script_path = workspace_dir / "python" / "main.py"

    # Compile the main.py script
    py_compile.compile(main_script_path)

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

    print("###########################################")
    print(f"score \t:\t {score}")
    print(f"time \t:\t {time_ms} ms")
    print(f"case \t:\t {case_no}")
    print("###########################################")


if __name__ == "__main__":
    main()
