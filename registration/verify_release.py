"""Check that the released code and data are byte-identical to the versions registered on OSF.

The registered versions are identified by git tree hashes (registration/registered_trees.txt) at two commits of the
author's private repository: 5bc6fad (OSF du2pr) and 3182bf1 (OSF 2jqhr). A git tree hash depends only on file names,
modes and contents, so it can be recomputed from the released files without the private history. Code changed after
the second registration only by adding options (all defaults unchanged); those changes are in
registration/src_reach_since_3182bf1.patch and are reversed here before hashing, as is the change between the two
registrations (src_reach_5bc6fad_to_3182bf1.patch). Requires git. Run from the repository root:

    python3 registration/verify_release.py
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "registration"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")


def git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def tree_hash(folder, reverse_patches=()):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "t"
        shutil.copytree(folder, work, ignore=IGNORE)
        git(["init", "-q"], work)
        for patch in reverse_patches:                  # patch paths are a/src/reach/...: strip three components
            git(["apply", "-R", "-p3", str(patch)], work)
        git(["add", "-A"], work)
        return git(["write-tree"], work)


def main():
    rows = [l.split() for l in (REG / "registered_trees.txt").read_text().splitlines() if l and not l.startswith("#")]
    since = REG / "src_reach_since_3182bf1.patch"
    between = REG / "src_reach_5bc6fad_to_3182bf1.patch"
    ok = True
    for path, commit, expected in rows:
        patches = []
        if path == "src/reach":
            patches = [since] if commit == "3182bf1" else [since, between]
        got = tree_hash(ROOT / path, patches)
        same = got == expected
        ok &= same
        how = " (after reversing the listed code changes)" if patches else ""
        print(f"{'OK  ' if same else 'FAIL'} {path} = registered {commit}{how}")
    print("all released files match the registered versions" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
