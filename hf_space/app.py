"""Hugging Face Space entry point for the OAVC interactive brush demo (SD3.5-medium).

The OAVC code is fetched from the public GitHub repo at start-up, so this Space only needs
app.py, requirements.txt and README.md. On ZeroGPU hardware the edit function runs inside
@spaces.GPU; on a dedicated GPU (or locally) the decorator is a no-op.

Environment variables (optional):
  HF_TOKEN       token with access to the gated stabilityai/stable-diffusion-3.5-medium (add as a Space secret)
  SD3_MODEL_ID   model id or local path (default: stabilityai/stable-diffusion-3.5-medium)
  OAVC_REPO_DIR  use an existing OAVC checkout instead of cloning
  OAVC_REF       git branch/tag to clone (default: main)
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.environ.get("OAVC_REPO_DIR") or os.path.join(HERE, "OAVC")
if not os.path.isdir(os.path.join(REPO_DIR, "oavc")):
    subprocess.run(["git", "clone", "--depth", "1", "--branch", os.environ.get("OAVC_REF", "main"),
                    "https://github.com/wwj-code/OAVC.git", REPO_DIR], check=True)
sys.path.insert(0, os.path.join(REPO_DIR, "demo"))
sys.path.insert(0, os.path.join(REPO_DIR, "oavc"))

try:
    import spaces  # available on Hugging Face Spaces; GPU() is a no-op outside ZeroGPU

    GPU = spaces.GPU(duration=120)
except ImportError:  # running locally without the `spaces` package
    def GPU(fn):
        return fn

from gradio_brush import OAVCEditor, create_demo  # noqa: E402

MODEL_ID = os.environ.get("SD3_MODEL_ID", "stabilityai/stable-diffusion-3.5-medium")
editor = OAVCEditor("SD3", sd3_ckpt=MODEL_ID, flux_ckpt=None, device="cuda")

_edit = editor.edit


@GPU
def run_edit(*args):
    return _edit(*args)


editor.edit = run_edit  # create_demo wires the Run button to editor.edit

demo = create_demo(editor)

if __name__ == "__main__":
    demo.queue(max_size=16).launch()
