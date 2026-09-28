"""Make the speaches build CPU-only, at build time, before `uv lock`.

Upstream's dependencies pull GPU software even into the image built on a plain
Ubuntu base: `kokoro-onnx[gpu]`, an override forcing `onnxruntime-gpu` on
x86_64, and CUDA builds of PyTorch via pyannote-audio. The result was a 4.9 GB
image with one 4.7 GB layer -- NVIDIA libraries for a machine with no NVIDIA
GPU, and a download that a flaky connection could not finish.

This rewrites pyproject.toml in the CI workspace only. The fork's own files are
left as upstream has them, so syncing with upstream never conflicts; if
upstream changes the lines this relies on, the assertions below fail the build
loudly instead of shipping a GPU image by accident.
"""
import pathlib
import re
import sys

path = pathlib.Path("pyproject.toml")
text = path.read_text(encoding="utf-8")

# 1. kokoro without its GPU extra.
if '"kokoro-onnx[gpu]' not in text:
    sys.exit("expected kokoro-onnx[gpu] in pyproject.toml; update cpu_only.py")
text = text.replace('"kokoro-onnx[gpu]', '"kokoro-onnx')

# 2. The x86_64 override to onnxruntime-gpu becomes plain onnxruntime.
pattern = re.compile(r'"onnxruntime-gpu[^"]*",\s*"onnxruntime([^";]*)[^"]*"')
if not pattern.search(text):
    sys.exit("expected the onnxruntime-gpu override in pyproject.toml; update cpu_only.py")
text = pattern.sub(lambda m: '"onnxruntime' + m.group(1).strip() + '"', text)

# 3. PyTorch from the CPU wheel index, not PyPI's CUDA default.
#
# `[tool.uv.sources]` only applies to direct dependencies, and torch arrives
# indirectly through pyannote-audio -- so a mapping alone was ignored and the
# lock still took PyPI's CUDA torch with fourteen NVIDIA packages. Declaring
# torch and torchaudio directly is what lets the mapping below take effect.
marker = "dependencies = [\n"
if text.count(marker) < 1:
    sys.exit("expected a dependencies list in pyproject.toml; update cpu_only.py")
text = text.replace(marker, marker + '    "torch",\n    "torchaudio",\n', 1)
if "[tool.uv.sources]" in text:
    sys.exit("pyproject.toml already has [tool.uv.sources]; merge into it in cpu_only.py")
text += """
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
torchaudio = [{ index = "pytorch-cpu" }]
"""

if "onnxruntime-gpu" in text:
    sys.exit("onnxruntime-gpu is still referenced after patching")

path.write_text(text, encoding="utf-8")
print("pyproject.toml patched for a CPU-only build")
