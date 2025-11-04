from pathlib import Path

tmp_dir = Path.home() / "tmp" / "cppcompile_windows"
(Path.home() / "tmp").mkdir(exist_ok=True)
tmp_dir.mkdir(exist_ok=True)
