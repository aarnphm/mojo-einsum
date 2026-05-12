"""Build a PEP 503 simple index for moeinsum artifacts."""

from __future__ import annotations

import argparse
import hashlib
import html
import re
import shutil
import sys
from pathlib import Path

PROJECT_NAMES = ("moeinsum",)
WHEEL_NAME_RE = re.compile(r"^(?P<dist>[^-]+(?:-[^-]+)*?)-\d.+\.whl$")
SDIST_NAME_RE = re.compile(r"^(?P<dist>[^-]+(?:-[^-]+)*?)-\d.+\.tar\.gz$")


def normalize(name: str) -> str:
  return re.sub(r"[-_.]+", "-", name).lower()


def detect_distribution(file_name: str) -> str | None:
  expected = {normalize(name) for name in PROJECT_NAMES}
  for pattern in (WHEEL_NAME_RE, SDIST_NAME_RE):
    match = pattern.match(file_name)
    if match is None:
      continue
    candidate = normalize(match.group("dist"))
    if candidate in expected:
      return candidate
  return None


def sha256_of(path: Path) -> str:
  hasher = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(65536), b""):
      hasher.update(chunk)
  return hasher.hexdigest()


def write_project_index(project_dir: Path, project_name: str, files: list[Path], commit: str) -> None:
  project_dir.mkdir(parents=True, exist_ok=True)
  links: list[str] = []
  for source in sorted(files):
    target = project_dir / source.name
    if target.resolve() != source.resolve():
      shutil.copy2(source, target)
    digest = sha256_of(target)
    href = f"{html.escape(target.name)}#sha256={digest}"
    links.append(f'    <a href="{href}">{html.escape(target.name)}</a><br/>')
  body = "\n".join(links)

  (project_dir / "index.html").write_text(
    "<!DOCTYPE html>\n"
    "<html>\n"
    f"  <head><title>Links for {html.escape(project_name)}</title></head>\n"
    "  <body>\n"
    f"    <h1>Links for {html.escape(project_name)} ({html.escape(commit)})</h1>\n"
    f"{body}\n"
    "  </body>\n"
    "</html>\n",
    encoding="utf-8",
  )


def write_root_index(simple_dir: Path, project_dirs: list[str]) -> None:
  links = "\n".join(f'    <a href="{html.escape(name)}/">{html.escape(name)}</a><br/>' for name in project_dirs)
  (simple_dir / "index.html").write_text(
    "<!DOCTYPE html>\n"
    "<html>\n"
    "  <head><title>Simple index</title></head>\n"
    "  <body>\n"
    "    <h1>Simple index</h1>\n"
    f"{links}\n"
    "  </body>\n"
    "</html>\n",
    encoding="utf-8",
  )


def write_landing(out_dir: Path, commit: str, commit_url: str) -> None:
  (out_dir / "index.html").write_text(
    "<!DOCTYPE html>\n"
    "<html>\n"
    "  <head><title>mojo-einsum development index</title></head>\n"
    "  <body>\n"
    "    <h1>mojo-einsum development index</h1>\n"
    f'    <p>Built from commit <a href="{html.escape(commit_url)}">{html.escape(commit)}</a>.</p>\n'
    "    <pre>uv pip install moeinsum --index-url https://aarnphm.github.io/mojo-einsum/simple/ "
    "--extra-index-url https://aarnphm.github.io/mohaus/simple/ "
    "--extra-index-url https://whl.modular.com/nightly/simple/ "
    "--extra-index-url https://pypi.org/simple --prerelease allow</pre>\n"
    '    <p><a href="simple/">PEP 503 index</a></p>\n'
    "  </body>\n"
    "</html>\n",
    encoding="utf-8",
  )


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dist", required=True, type=Path, help="Source dir of *.whl + *.tar.gz")
  parser.add_argument("--out", required=True, type=Path, help="Output site dir")
  parser.add_argument("--commit", required=True, help="Source commit SHA")
  parser.add_argument("--commit-url", required=True, help="URL pointing at the commit")
  parser.add_argument("--prune", action="store_true", help="Wipe --out before writing")
  args = parser.parse_args(argv)

  dist_dir: Path = args.dist
  out_dir: Path = args.out
  if args.prune and out_dir.exists():
    shutil.rmtree(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  simple_dir = out_dir / "simple"
  simple_dir.mkdir(parents=True, exist_ok=True)

  buckets: dict[str, list[Path]] = {normalize(name): [] for name in PROJECT_NAMES}
  for path in sorted(dist_dir.rglob("*")):
    if not path.is_file():
      continue
    distribution = detect_distribution(path.name)
    if distribution is None:
      continue
    buckets[distribution].append(path)

  populated: list[str] = []
  for project_name in PROJECT_NAMES:
    key = normalize(project_name)
    files = buckets.get(key, [])
    if not files:
      print(f"warning: no artifacts for {project_name}", file=sys.stderr)
      continue
    project_dir = simple_dir / project_name
    write_project_index(project_dir, project_name, files, args.commit)
    populated.append(project_name)

  if not populated:
    print("error: no artifacts found in dist directory", file=sys.stderr)
    return 1

  write_root_index(simple_dir, populated)
  write_landing(out_dir, args.commit, args.commit_url)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
