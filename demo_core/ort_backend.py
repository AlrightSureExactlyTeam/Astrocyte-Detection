"""ONNX Runtime backend for CPSAM inside the V3.3-B.2 pipeline.

Why this exists: the accepted CPSAM checkpoint runs under torch in bfloat16 by
default. On CPUs without AVX512-BF16 (all consumer chips, shared Cloud vCPUs)
bf16 GEMM is emulated and ~24x slower than fp32, while the torch fp32 weights
need ~2x the RAM. The ONNX export of the same weights runs in fp32 under
onnxruntime with bit-identical masks (verified against the torch fp32 path)
at a fraction of the time and memory. The checkpoint, network architecture,
and inference algorithm are unchanged; only the execution runtime differs.
"""
from __future__ import annotations

import os
from pathlib import Path


class OrtNet:
    """Drop-in replacement for the CPSAM torch net used by cellpose.core._forward.

    The ONNX graph was exported from the accepted CPSAM checkpoint in fp32 with
    a dynamic batch axis; outputs are (flows, style) exactly like the torch net.
    """

    def __init__(self, onnx_path: str | Path):
        import onnxruntime as ort
        import torch

        self._torch = torch
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def eval(self):
        return self

    def __call__(self, X):
        torch = self._torch
        x = X.detach().cpu().to(torch.float32).numpy()
        flows, style = self.sess.run(["flows", "style"], {"input": x})
        return torch.from_numpy(flows), torch.from_numpy(style)


def build_ort_cellpose(onnx_path: str | Path):
    """Return a CellposeModel shell whose net is the ORT session.

    The shell never materializes the torch CPSAM weights, which keeps the
    Cloud memory profile at the ONNX level (~1.2 GiB instead of ~2.3 GiB).
    Only the attributes used by CellposeModel.eval -> _run_net/_compute_masks
    are populated (net, device, backbone, gpu, pretrained_model).
    """
    import torch
    from cellpose import models

    shell = models.CellposeModel.__new__(models.CellposeModel)
    shell.net = OrtNet(onnx_path)
    shell.device = torch.device("cpu")
    shell.gpu = False
    shell.backbone = "sam_vitl"
    shell.pretrained_model = str(onnx_path)
    return shell


def use_ort_backend(model) -> bool:
    """Backend selection: explicit env override wins; Cloud defaults to ORT.

    ASTROCYTE_CPSAM_BACKEND=ort|torch forces the backend locally. On Streamlit
    Cloud the ORT backend is the default whenever the registry declares the
    CPSAM ONNX artifact, because the torch path cannot run within Cloud limits.
    """
    from .weights import is_streamlit_cloud

    explicit = os.environ.get("ASTROCYTE_CPSAM_BACKEND", "").strip().lower()
    if explicit in {"ort", "torch"}:
        return explicit == "ort"
    return is_streamlit_cloud() and bool(
        model.deployment.get("cpsam_onnx_part_url_template_env")
    )
