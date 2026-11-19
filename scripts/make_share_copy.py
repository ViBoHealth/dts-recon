"""Make the shared copy of dts-recon for collaborators, without the real data or anything derived from it.

The copy is taken from a commit (default HEAD), not from the working tree, so uncommitted files never get in.
Every tracked file must match either SHARE or KEEP_OUT; a new file that matches neither stops the copy, so
each new path is a conscious decision. After writing, the copy is checked for leaks (excluded paths, home
directory paths, unexpected binaries, large files).

    python scripts/make_share_copy.py <target_dir> [--ref HEAD]

<target_dir> is the working tree of the shared repository (e.g. ../dts-recon-shared). Everything in it except
.git is replaced. Commit and push there yourself.
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Material from real data (the 2018 emitter-array scans) and internal notes. Checked first.
KEEP_OUT = [
    "reports/real_data_*",
    "reports/figures/real_data_*/*",
    "configs/real/*",
    "scripts/real_v*.py",
    "scripts/poc_intraoral.py",
    "scripts/make_share_copy.py",
    "docs/decisions.md",
    "docs/STATUS.md",
    "CLAUDE.md",
    "data/*",
]

SHARE = [
    "README.md",
    "pyproject.toml",
    ".gitignore",
    "src/dtsrecon/*",
    "tests/*",
    "third_party/leap/*",
    "configs/*",
    "docs/*",
    "scripts/*",
    "reports/*",
]

# Binary files allowed in the copy (simulation figures only; real-data figures are excluded above).
BINARY_OK = [".png"]
MAX_BYTES = 2_000_000

README_LINE = (
    "Shared copy of the project, made with `scripts/make_share_copy.py` from commit {sha}. It leaves out the "
    "real-data material (2018 emitter-array scans: reports, figures, configs, scripts) and internal notes "
    "(`CLAUDE.md`, `docs/decisions.md`, `docs/STATUS.md`); references to them in the docs do not resolve here. "
    "Project brief: `docs/PROJECT_BRIEF.md`. Usage: `docs/USAGE.md`. Packaging plan: `docs/PACKAGING.md`."
)


def git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True).stdout


def match(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, p) for p in patterns)


def tracked(ref: str) -> list[tuple[str, str]]:
    """(mode, path) of every file in ref."""
    out = []
    for entry in git("ls-tree", "-r", "-z", ref).split(b"\0"):
        if not entry:
            continue
        meta, path = entry.split(b"\t", 1)
        mode, kind, _ = meta.split()
        if kind != b"blob" or mode == b"120000":
            raise SystemExit(f"unsupported entry (submodule or symlink): {path.decode()}")
        out.append((mode.decode(), path.decode()))
    return out


def select(files: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], list[str]]:
    share, out, unknown = [], [], []
    for mode, path in files:
        if match(path, KEEP_OUT):
            out.append(path)
        elif match(path, SHARE):
            share.append((mode, path))
        else:
            unknown.append(path)
    if unknown:
        raise SystemExit("files in neither SHARE nor KEEP_OUT (add them to one list):\n  " + "\n  ".join(unknown))
    return share, out


def scrub(path: str, data: bytes, sha: str) -> bytes:
    """Home-directory paths become ~; the README's status line points at what the copy holds."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    text = text.replace(str(Path.home()), "~")
    if path == "README.md":
        lines = text.split("\n")
        idx = next((i for i, ln in enumerate(lines) if ln.startswith("Project brief:")), None)
        note = README_LINE.format(sha=sha)
        if idx is None:
            lines.insert(1, "\n" + note)
        else:
            lines[idx] = note
        text = "\n".join(lines)
    return text.encode("utf-8")


def check(target: Path) -> list[str]:
    problems = []
    home = str(Path.home()).encode()
    user = Path.home().name.encode()
    for f in sorted(p for p in target.rglob("*") if p.is_file() and ".git" not in p.relative_to(target).parts):
        rel = f.relative_to(target).as_posix()
        data = f.read_bytes()
        if match(rel, KEEP_OUT):
            problems.append(f"excluded path present: {rel}")
        if len(data) > MAX_BYTES:
            problems.append(f"larger than {MAX_BYTES} bytes: {rel}")
        try:
            data.decode("utf-8")
            if home in data or user in data:
                problems.append(f"home directory or user name in: {rel}")
        except UnicodeDecodeError:
            if f.suffix.lower() not in BINARY_OK:
                problems.append(f"unexpected binary file: {rel}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", type=Path)
    ap.add_argument("--ref", default="HEAD")
    a = ap.parse_args()

    target = a.target.resolve()
    if target == REPO or REPO in target.parents:
        raise SystemExit("target must be outside this repository")
    sha = git("rev-parse", "--short", a.ref).decode().strip()
    share, out = select(tracked(a.ref))

    target.mkdir(parents=True, exist_ok=True)
    for p in target.iterdir():
        if p.name != ".git":
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    for mode, path in share:
        dst = target / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(scrub(path, git("cat-file", "blob", f"{a.ref}:{path}"), sha))
        if mode == "100755":
            dst.chmod(0o755)

    problems = check(target)
    print(f"{a.ref} ({sha}): {len(share)} files copied to {target}, {len(out)} kept out:")
    for p in out:
        print(f"  - {p}")
    if problems:
        print("LEAK CHECK FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    print("leak check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
