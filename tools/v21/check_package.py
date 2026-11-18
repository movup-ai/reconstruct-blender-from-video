#!/usr/bin/env python3
"""白模交付包上传前自检（对应《交付规范》第 1、2 节）。

用法（Windows 用 py，Linux/macOS 用 python3）：
    py check_package.py the_greatest_showman-1.zip comparison.mp4
    py check_package.py D:\交付\the_greatest_showman-1\v1      # 版本目录，自动找 zip 和 comparison.mp4

Windows 上也可以把版本目录直接拖到同目录的 check_package.bat 上。

全部检查通过时，在 zip 旁边生成 {包名}.zip.sha256，并打印 manifest_v{n}.csv 需要的
zip_sha256、zip_size、comparison_sha256。有任何一项不通过时，退出码为 1，并且
不会生成 .zip.sha256（如果旁边留着旧的，会删掉，避免把没通过的包传上去）。

只依赖 Python 3.9+ 标准库。这里只检查文件结构；视频能否完整解码由收货方验收时检查。
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
# .gz 不算：meta/verification/ 下的 *.json.gz 是正常的压缩数据文件，不是嵌套包
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
        print(f"  [通过]   {item}" + (f"：{detail}" if detail else ""))

    def fail(self, item: str, detail: str) -> None:
        self.failed += 1
        print(f"  [不通过] {item}：{detail}")

    def note(self, item: str, detail: str) -> None:
        print(f"  [提示]   {item}：{detail}")

    def check(self, item: str, problems: list[str], ok_detail: str = "") -> None:
        if problems:
            more = f"，另有 {len(problems) - 3} 处" if len(problems) > 3 else ""
            self.fail(item, "；".join(problems[:3]) + more)
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
    """检查 mp4 的 box 结构，返回 (错误描述, (宽, 高))，没有错误时描述为空串。"""
    if len(data) < 16:
        return "文件过小，不是有效的 mp4", (0, 0)
    top = set()
    total = 0
    for typ, _bs, be in _iter_boxes(data, 0, len(data)):
        top.add(typ)
        total = be
    for required in (b"ftyp", b"moov", b"mdat"):
        if required not in top:
            return f"缺少 {required.decode()} box", (0, 0)
    if total < len(data) - 8:
        return "box 长度与文件大小不符，文件可能被截断", (0, 0)

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
        return "找不到视频轨的尺寸信息", (0, 0)
    return "", (size[0], size[1])


def resolution_ok(width: int, height: int) -> bool:
    return width == REQUIRED_WIDTH and height > 0 and height % HEIGHT_UNIT == 0


# ---------------------------------------------------------------- blend


def blend_header_error(raw: bytes) -> str:
    """raw 是 .blend 的开头若干字节，返回错误描述，有效时返回空串。"""
    if raw.startswith(GZIP_MAGIC):
        try:
            raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read(len(BLENDER_MAGIC))
        except (OSError, EOFError) as exc:
            return f"gzip 解压失败：{exc}"
    elif raw.startswith(ZSTD_MAGIC):
        try:
            from compression import zstd  # Python 3.14+
        except ImportError:
            return ""  # 旧版 Python 解不了 zstd，文件头交给收货方检查
        try:
            raw = zstd.ZstdDecompressor().decompress(raw, max_length=len(BLENDER_MAGIC))
        except Exception as exc:
            return f"zstd 解压失败：{exc}"
    if not raw.startswith(BLENDER_MAGIC):
        return f"文件头不是 BLENDER：{raw[:12]!r}"
    return ""


# ---------------------------------------------------------------- 主流程


def check(zip_path: Path, comparison_path: Path) -> int:
    report = Report()
    package = zip_path.name[:-4] if zip_path.name.lower().endswith(".zip") else zip_path.stem
    print(f"检查 {zip_path.name}（包名 {package}）")

    # 1. 包名与格式
    if zip_path.suffix != ".zip":
        report.fail("压缩格式", f"扩展名是 {zip_path.suffix or '无'}，只接受 .zip")
        return finish(report, zip_path, comparison_path)
    report.check(
        "包名",
        [] if PACKAGE_NAME_RE.match(package) else
        [f"{package!r} 不符合 {{片名}}-{{序号}}，只能用小写字母、数字、_、-"],
        package,
    )

    try:
        zf = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError) as exc:
        report.fail("zip 能否打开", str(exc))
        return finish(report, zip_path, comparison_path)

    with zf:
        broken = zf.testzip()
        report.check("zip 完整性", [f"条目损坏：{broken}"] if broken else [], "所有条目 CRC 正确")

        infos = zf.infolist()
        names = [i.filename for i in infos]
        files = [i for i in infos if not i.is_dir()]

        # 2.2 反斜杠
        report.check(
            "路径分隔符",
            [f"含反斜杠：{n}" for n in names if "\\" in n],
            "全部使用 /",
        )

        # 2.3 结构
        roots = {n.split("/")[0] for n in names}
        wrap = []
        if len(roots) == 1 and "comparison" not in roots:
            wrap = [f"外面套了一层 {next(iter(roots))}/ 目录，请直接压缩目录里的内容"]
        report.check("没有套层目录", wrap)

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
            "必需文件",
            [f"缺少 {r}" for r in required if r not in name_set],
            "、".join(required),
        )

        # 2.4 内容
        report.check(
            "中文文件名使用 UTF-8 记录",
            [
                f"{i.filename}（文件名编码不是 UTF-8，在其他系统上会乱码）"
                for i in infos
                if not i.flag_bits & UTF8_FLAG and any(ord(c) > 127 for c in i.filename)
            ],
        )
        report.check("路径没有空格", [n for n in names if " " in n])
        report.check(
            "没有嵌套压缩包",
            [i.filename for i in files if i.filename.lower().endswith(NESTED_SUFFIXES)],
        )
        report.check(
            "没有系统垃圾文件",
            [n for n in names if JUNK_PARTS.intersection(n.replace("\\", "/").split("/"))],
        )

        mp4_problems, res_notes = [], []
        for info in files:
            if not info.filename.lower().endswith(".mp4"):
                continue
            err, (w, h) = parse_mp4(zf.read(info))
            if err:
                mp4_problems.append(f"{info.filename}：{err}")
            elif not resolution_ok(w, h):
                res_notes.append(f"{info.filename} {w}x{h}")
        report.check("mp4 结构", mp4_problems)
        if res_notes:
            report.note("分辨率不符 1920x(1080 的整数倍)，不会退回但会被标记", "；".join(res_notes))

        blend_problems = []
        blends = [i for i in files if i.filename.lower().endswith(".blend")]
        for info in blends:
            with zf.open(info) as fh:
                err = blend_header_error(fh.read(4096))
            if err:
                blend_problems.append(f"{info.filename}：{err}")
        report.check("Blender 文件头", blend_problems, f"{len(blends)} 个 .blend")

        # 对比视频一致
        if not comparison_path.is_file():
            report.fail("对比视频", f"找不到 {comparison_path}")
            return finish(report, zip_path, comparison_path)
        if comparison_path.name != "comparison.mp4":
            report.note("对比视频文件名", f"上传到 R2 时要命名为 comparison.mp4（现在是 {comparison_path.name}）")
        err, _ = parse_mp4(comparison_path.read_bytes())
        report.check("对比视频 mp4 结构", [err] if err else [])
        if "comparison/comparison.mp4" in name_set:
            outer = sha256_file(comparison_path)
            inner = sha256_entry(zf, "comparison/comparison.mp4")
            report.check(
                "包内外对比视频一致",
                [] if outer == inner else [f"包外 {outer[:16]}… 与包内 {inner[:16]}… 不同"],
                outer[:16] + "…",
            )

    return finish(report, zip_path, comparison_path)


def finish(report: Report, zip_path: Path, comparison_path: Path) -> int:
    sha_path = zip_path.with_name(zip_path.name + ".sha256")
    if report.failed:
        print(f"\n结果：{report.failed} 项不通过，请修改后重新打包再检查。")
        if sha_path.exists():
            sha_path.unlink()
            print(f"已删除旧的 {sha_path.name}，避免和没通过检查的包一起上传。")
        return 1

    zip_sha = sha256_file(zip_path)
    sha_path.write_text(f"{zip_sha}  {zip_path.name}\n", encoding="ascii", newline="\n")
    print("\n结果：全部通过。")
    print(f"已生成 {sha_path.name}")
    print("manifest_v{n}.csv 需要的字段：")
    print(f"  zip_sha256        {zip_sha}")
    print(f"  zip_size          {zip_path.stat().st_size}")
    print(f"  comparison_sha256 {sha256_file(comparison_path)}")
    return 0


USAGE = (
    "用法：py check_package.py {包名}.zip comparison.mp4\n"
    "  或：py check_package.py {版本目录}"
)


def resolve_folder(folder: Path) -> tuple[Path, Path] | None:
    """版本目录里应该恰好有一个 .zip 和一个 comparison.mp4。"""
    zips = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".zip")
    comparison = folder / "comparison.mp4"
    if len(zips) != 1 or not comparison.is_file():
        found = "、".join(p.name for p in sorted(folder.iterdir())) or "空目录"
        print(f"{folder} 里应该恰好有一个 .zip 和一个 comparison.mp4，实际是：{found}")
        return None
    return zips[0], comparison


def main(argv: list[str]) -> int:
    # Windows 控制台输出被重定向到文件时用的是本地编码（如 GBK），遇到编不了的字符
    # 也不能让检查中途崩掉
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    # cmd 里 "D:\交付\v1\" 这种以反斜杠结尾的带引号路径，会把结尾的引号当成普通字符传进来
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
