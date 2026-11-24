#!/usr/bin/env python3
r"""Pre-upload white-model package checks (delivery specification sections 1 and 2).

Usage (Windows: py; Linux/macOS: python3):
    py check_package.py the_greatest_showman-1.zip comparison.mp4
    py check_package.py D:\deliveries\the_greatest_showman-1\v1

The second form locates the zip and comparison.mp4 in a version directory.
On Windows, you can also drag the version directory onto check_package.bat.

On success, writes {package}.zip.sha256 beside the zip and prints the zip_sha256,
zip_size and comparison_sha256 fields for manifest_v{n}.csv. A failed check exits
with code 1 and removes any old .zip.sha256 file to avoid uploading a failed package.

Requires only the Python 3.9+ standard library. Checks file structure; the recipient
must separately verify that the videos decode completely.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import re
import struct
import sys
import zipfile
from pathlib import Path

PACKAGE_NAME_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*-[0-9]+$")
# .gz is allowed: meta/verification/*.json.gz contains compressed data, not nested archives.
NESTED_SUFFIXES = (".zip", ".rar", ".7z")
JUNK_PARTS = {"__MACOSX", ".DS_Store", "Thumbs.db", "desktop.ini"}
UTF8_FLAG = 0x800

BLENDER_MAGIC = b"BLENDER"
GZIP_MAGIC = b"\x1f\x8b"
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
REQUIRED_WIDTH = 1920
HEIGHT_UNIT = 1080


class Report:
    def __init__(self) -> None:
        self.failed = 0

    def ok(self, item: str, detail: str = "") -> None:
        print(f"  [PASS]   {item}" + (f": {detail}" if detail else ""))

    def fail(self, item: str, detail: str) -> None:
        self.failed += 1
        print(f"  [FAIL] {item}: {detail}")

    def note(self, item: str, detail: str) -> None:
        print(f"  [NOTE]   {item}: {detail}")

    def check(self, item: str, problems: list[str], ok_detail: str = "") -> None:
        if problems:
            more = f", plus {len(problems) - 3} more" if len(problems) > 3 else ""
            self.fail(item, "; ".join(problems[:3]) + more)
        else:
            self.ok(item, ok_detail)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_entry(zf: zipfile.ZipFile, name: str) -> str:
    h = hashlib.sha256()
    with zf.open(name) as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ mp4


def _iter_boxes(data: bytes, start: int, end: int):
    i = start
    while i + 8 <= end:
        size = struct.unpack(">I", data[i : i + 4])[0]
        typ = data[i + 4 : i + 8]
        head = 8
        if size == 1:
            if i + 16 > end:
                return
            size = struct.unpack(">Q", data[i + 8 : i + 16])[0]
            head = 16
        elif size == 0:
            size = end - i
        if size < head:
            return
        yield typ, i + head, min(i + size, end)
        i += size


def parse_mp4(data: bytes) -> tuple[str, tuple[int, int]]:
    """Check MP4 boxes; return (error, (width, height)), with an empty error on success."""
    if len(data) < 16:
        return "File too small to be a valid MP4", (0, 0)
    top = set()
    total = 0
    for typ, _bs, be in _iter_boxes(data, 0, len(data)):
        top.add(typ)
        total = be
    for required in (b"ftyp", b"moov", b"mdat"):
        if required not in top:
            return f"Missing {required.decode()} box", (0, 0)
    if total < len(data) - 8:
        return "Box lengths do not match file size; the file may be truncated", (0, 0)

    size = [0, 0]

    def walk(s: int, e: int) -> None:
        for typ, bs, be in _iter_boxes(data, s, e):
            if typ in (b"moov", b"trak", b"mdia", b"minf", b"stbl"):
                walk(bs, be)
            elif typ == b"stsd":
                for t2, b2, _e2 in _iter_boxes(data, bs + 8, be):
                    if t2 in (b"avc1", b"hev1", b"hvc1", b"mp4v") and b2 + 28 <= len(data):
                        w, h = struct.unpack(">HH", data[b2 + 24 : b2 + 28])
                        if w and h and not size[0]:
                            size[0], size[1] = w, h

    walk(0, len(data))
    if not size[0]:
        return "Video track dimensions not found", (0, 0)
    return "", (size[0], size[1])


def resolution_ok(width: int, height: int) -> bool:
    return width == REQUIRED_WIDTH and height > 0 and height % HEIGHT_UNIT == 0


# ---------------------------------------------------------------- blend


def blend_header_error(raw: bytes) -> str:
    """Check the initial bytes of a .blend file; return an empty error on success."""
    if raw.startswith(GZIP_MAGIC):
        try:
            raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read(len(BLENDER_MAGIC))
        except (OSError, EOFError) as exc:
            return f"gzip decompression failed: {exc}"
    elif raw.startswith(ZSTD_MAGIC):
        try:
            from compression import zstd  # Python 3.14+
        except ImportError:
            return ""  # Older Python cannot decode zstd; defer header verification to the recipient.
        try:
            raw = zstd.ZstdDecompressor().decompress(raw, max_length=len(BLENDER_MAGIC))
        except Exception as exc:
            return f"zstd decompression failed: {exc}"
    if not raw.startswith(BLENDER_MAGIC):
        return f"File header is not BLENDER: {raw[:12]!r}"
    return ""


# ---------------------------------------------------------------- main flow


def check(zip_path: Path, comparison_path: Path) -> int:
    report = Report()
    package = zip_path.name[:-4] if zip_path.name.lower().endswith(".zip") else zip_path.stem
    print(f"Checking {zip_path.name} (package {package})")

    # 1. Package name and format
    if zip_path.suffix != ".zip":
        report.fail("Archive format", f"Extension is {zip_path.suffix or 'none'}; only .zip is accepted")
        return finish(report, zip_path, comparison_path)
    report.check(
        "Package name",
        [] if PACKAGE_NAME_RE.match(package) else
        [f"{package!r} does not match {{title}}-{{number}}; use lowercase letters, digits, _ and - only"],
        package,
    )

    try:
        zf = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError) as exc:
        report.fail("ZIP opens", str(exc))
        return finish(report, zip_path, comparison_path)

    with zf:
        broken = zf.testzip()
        report.check("ZIP integrity", [f"Corrupt entry: {broken}"] if broken else [], "All entry CRCs are valid")

        infos = zf.infolist()
        names = [i.filename for i in infos]
        files = [i for i in infos if not i.is_dir()]

        # 2.2 Backslashes
        report.check(
            "Path separators",
            [f"Contains backslashes: {n}" for n in names if "\\" in n],
            "All paths use /",
        )

        # 2.3 structure
        roots = {n.split("/")[0] for n in names}
        wrap = []
        if len(roots) == 1 and "comparison" not in roots:
            wrap = [f"Unexpected wrapper directory {next(iter(roots))}/; archive its contents directly"]
        report.check("No wrapper directory", wrap)

        required = [
            "comparison/comparison.mp4",
            "target/original.mp4",
            "white_model/blendervid.mp4",
            f"model_3d/{package}.blend",
            "model_3d/reconstruction/scene.json",
            "meta/meta.json",
        ]
        name_set = set(names)
        report.check(
            "Required files",
            [f"Missing {r}" for r in required if r not in name_set],
            ", ".join(required),
        )

        # 2.4 Contents
        report.check(
            "Non-ASCII filenames use UTF-8",
            [
                f"{i.filename} (filename is not encoded as UTF-8 and may display incorrectly on other systems)"
                for i in infos
                if not i.flag_bits & UTF8_FLAG and any(ord(c) > 127 for c in i.filename)
            ],
        )
        report.check("No spaces in paths", [n for n in names if " " in n])
        report.check(
            "No nested archives",
            [i.filename for i in files if i.filename.lower().endswith(NESTED_SUFFIXES)],
        )
        report.check(
            "No system junk files",
            [n for n in names if JUNK_PARTS.intersection(n.replace("\\", "/").split("/"))],
        )

        mp4_problems, res_notes = [], []
        for info in files:
            if not info.filename.lower().endswith(".mp4"):
                continue
            err, (w, h) = parse_mp4(zf.read(info))
            if err:
                mp4_problems.append(f"{info.filename}: {err}")
            elif not resolution_ok(w, h):
                res_notes.append(f"{info.filename} {w}x{h}")
        report.check("mp4 structure", mp4_problems)
        if res_notes:
            report.note("Resolution differs from 1920x(a multiple of 1080); flagged without rejecting the package", "; ".join(res_notes))

        blend_problems = []
        blends = [i for i in files if i.filename.lower().endswith(".blend")]
        for info in blends:
            with zf.open(info) as fh:
                err = blend_header_error(fh.read(4096))
            if err:
                blend_problems.append(f"{info.filename}: {err}")
        report.check("Blender file headers", blend_problems, f"{len(blends)} .blend files")

        # Comparison videos match
        if not comparison_path.is_file():
            report.fail("Comparison video", f"Cannot find {comparison_path}")
            return finish(report, zip_path, comparison_path)
        if comparison_path.name != "comparison.mp4":
            report.note("Comparison video filename", f"Name it comparison.mp4 when uploading to R2 (currently {comparison_path.name})")
        err, _ = parse_mp4(comparison_path.read_bytes())
        report.check("Comparison video mp4 structure", [err] if err else [])
        if "comparison/comparison.mp4" in name_set:
            outer = sha256_file(comparison_path)
            inner = sha256_entry(zf, "comparison/comparison.mp4")
            report.check(
                "Internal and external comparison videos match",
                [] if outer == inner else [f"External {outer[:16]}… differs from internal {inner[:16]}…"],
                outer[:16] + "…",
            )

    return finish(report, zip_path, comparison_path)


def finish(report: Report, zip_path: Path, comparison_path: Path) -> int:
    sha_path = zip_path.with_name(zip_path.name + ".sha256")
    if report.failed:
        print(f"\nResult: {report.failed} failed checks. Fix, repackage and check again.")
        if sha_path.exists():
            sha_path.unlink()
            print(f"Removed old {sha_path.name} to avoid uploading it with a failed package.")
        return 1

    zip_sha = sha256_file(zip_path)
    sha_path.write_text(f"{zip_sha}  {zip_path.name}\n", encoding="ascii", newline="\n")
    print("\nResult: all checks passed.")
    print(f"Created {sha_path.name}")
    print("Fields for manifest_v{n}.csv:")
    print(f"  zip_sha256        {zip_sha}")
    print(f"  zip_size          {zip_path.stat().st_size}")
    print(f"  comparison_sha256 {sha256_file(comparison_path)}")
    return 0


USAGE = (
    "Usage: py check_package.py {package}.zip comparison.mp4\n"
    "  or: py check_package.py {version_directory}"
)


def resolve_folder(folder: Path) -> tuple[Path, Path] | None:
    """A version directory must contain exactly one .zip and a comparison.mp4."""
    zips = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".zip")
    comparison = folder / "comparison.mp4"
    if len(zips) != 1 or not comparison.is_file():
        found = ", ".join(p.name for p in sorted(folder.iterdir())) or "empty directory"
        print(f"{folder} must contain exactly one .zip and a comparison.mp4; found: {found}")
        return None
    return zips[0], comparison


def main(argv: list[str]) -> int:
    # Redirected Windows console output can use a local encoding such as GBK.
    # Unencodable characters must not interrupt the check.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    # cmd can pass a trailing quote literally in a quoted path ending with a backslash.
    args = [Path(a.strip().strip('"')) for a in argv]
    if len(args) == 1 and args[0].is_dir():
        pair = resolve_folder(args[0])
        if pair is None:
            return 2
        return check(*pair)
    if len(args) == 2:
        return check(args[0], args[1])
    print(USAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
