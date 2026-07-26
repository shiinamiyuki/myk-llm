import os
import sys
from typing import *
import random

def find_all_files(directory: str, predicate: Callable[[str], bool]) -> List[str]:
    result = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if predicate(file):
                result.append(os.path.join(root, file))
    return result

USAGE= """"Usage: python aggregate.py <directory> <extensions> <output>\n"
extensions: a comma-separated list of file extensions to include (e.g. .py,.txt)\n"
"""

if __name__ == "__main__":
    assert len(sys.argv) == 4, USAGE
    directory = sys.argv[1]
    extensions = sys.argv[2].split(",")
    output = sys.argv[3]
    files = find_all_files(directory, lambda file: any(file.endswith(ext) for ext in extensions))
    random.shuffle(files)
    with open(output, "w", encoding="utf-8") as f:
        for file in files:
            try:
                with open(file, "r", encoding="utf-8") as g:
                    s = g.read()
                    f.write(s)
            except Exception as e:
                print(f"Error reading file {file}: {e}", file=sys.stderr)
    print(f"Aggregated {len(files)} files into {output}")
    # prints all file names that were aggregated
    for file in files:
        print(file)