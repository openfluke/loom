# src/welvet/utils.py
"""
welvet - LOOM Python Bindings (M-POLY-VTD Architecture)

Wraps the Loom C ABI for Python access. Built on the polymorphic
volumetric tiled-tensor dispatcher (M-POLY-VTD) with support for
21 numerical types and WebGPU acceleration.
"""

import sys
import json
import ctypes
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional
from importlib.resources import files

PKG_DIR = files("welvet")
_RTLD_GLOBAL = getattr(ctypes, "RTLD_GLOBAL", 0)


def _lib_path() -> Path:
    """Determine the correct native library path for the current platform."""
    plat = sys.platform
    arch = platform.machine().lower()

    # Map Python machine names to Go GOARCH names (used by the builder)
    arch_map = {
        "x86_64": "amd64",
        "amd64":  "amd64",
        "aarch64": "arm64",
        "arm64":   "arm64",
        "armv7l":  "arm",
        "i686":    "386",
        "i386":    "386",
    }
    arch_key = arch_map.get(arch, arch)

    if plat.startswith("linux"):
        lib_name = "welvet.so"
        platform_dir = f"linux_{arch_key}"
    elif plat == "darwin":
        lib_name = "welvet.dylib"
        platform_dir = f"darwin_{arch_key}"
        candidate = PKG_DIR / platform_dir / lib_name
        if not Path(candidate).is_file():
            platform_dir = "darwin_universal"
    elif plat.startswith("win"):
        lib_name = "welvet.dll"
        platform_dir = f"windows_{arch_key}"
    else:
        raise RuntimeError(f"Unsupported platform: {plat} ({arch})")

    lib_path = PKG_DIR / platform_dir / lib_name
    if not Path(lib_path).is_file():
        raise FileNotFoundError(
            f"LOOM native library not found at {lib_path}\n"
            f"Platform: {plat}, Architecture: {arch_key}\n"
            f"Expected directory: {platform_dir}"
        )
    return Path(lib_path)


# Load the native library
_LIB = ctypes.CDLL(str(_lib_path()), mode=_RTLD_GLOBAL)


def _sym(name: str):
    """Get a symbol from the loaded library, returning None if not found."""
    try:
        return getattr(_LIB, name)
    except AttributeError:
        return None


def _parse_json(ptr) -> Any:
    """Read a C string pointer as UTF-8 JSON."""
    if not ptr:
        return None
    if isinstance(ptr, int):
        raw = ctypes.string_at(ptr)
    else:
        raw = ctypes.cast(ptr, ctypes.c_char_p).value
    if not raw:
        return None
    return json.loads(raw.decode("utf-8", errors="replace"))


def _check(result: Any, ctx: str = "") -> Any:
    """Raise RuntimeError if result dict contains an 'error' key."""
    if isinstance(result, dict) and "error" in result:
        msg = result["error"]
        raise RuntimeError(f"{ctx}: {msg}" if ctx else msg)
    return result


def _to_cfloat_array(data: List[float]):
    """Convert a Python float list to a ctypes float array."""
    arr = (ctypes.c_float * len(data))(*data)
    return arr, len(data)


# ---------------------------------------------------------------------------
# C Function Bindings
# ---------------------------------------------------------------------------

# Memory management
_FreeLoomString = _sym("FreeLoomString")
if _FreeLoomString:
    _FreeLoomString.argtypes = [ctypes.c_char_p]

# Network lifecycle
_LoomBuildNetworkFromJSON = _sym("LoomBuildNetworkFromJSON")
if _LoomBuildNetworkFromJSON:
    _LoomBuildNetworkFromJSON.restype = ctypes.c_longlong
    _LoomBuildNetworkFromJSON.argtypes = [ctypes.c_char_p]

_LoomCreateNetwork = _sym("LoomCreateNetwork")
if _LoomCreateNetwork:
    _LoomCreateNetwork.restype = ctypes.c_longlong
    _LoomCreateNetwork.argtypes = [ctypes.c_char_p]

_LoomLoadUniversal = _sym("LoomLoadUniversal")
if _LoomLoadUniversal:
    _LoomLoadUniversal.restype = ctypes.c_longlong
    _LoomLoadUniversal.argtypes = [ctypes.c_char_p]

_LoomLoadUniversalDetailed = _sym("LoomLoadUniversalDetailed")
if _LoomLoadUniversalDetailed:
    _LoomLoadUniversalDetailed.restype = ctypes.c_char_p
    _LoomLoadUniversalDetailed.argtypes = [ctypes.c_char_p]

_LoomFreeNetwork = _sym("LoomFreeNetwork")
if _LoomFreeNetwork:
    _LoomFreeNetwork.argtypes = [ctypes.c_longlong]

_LoomGetNetworkInfo = _sym("LoomGetNetworkInfo")
if _LoomGetNetworkInfo:
    _LoomGetNetworkInfo.restype = ctypes.c_char_p
    _LoomGetNetworkInfo.argtypes = [ctypes.c_longlong]

_LoomGetMethodsJSON = _sym("LoomGetMethodsJSON")
if _LoomGetMethodsJSON:
    _LoomGetMethodsJSON.restype = ctypes.c_char_p
    _LoomGetMethodsJSON.argtypes = []

_LoomGetLayerTelemetry = _sym("LoomGetLayerTelemetry")
if _LoomGetLayerTelemetry:
    _LoomGetLayerTelemetry.restype = ctypes.c_char_p
    _LoomGetLayerTelemetry.argtypes = [ctypes.c_longlong, ctypes.c_int]

_LoomExtractDNA = _sym("LoomExtractDNA")
if _LoomExtractDNA:
    _LoomExtractDNA.restype = ctypes.c_char_p
    _LoomExtractDNA.argtypes = [ctypes.c_longlong]

_LoomCompareDNA = _sym("LoomCompareDNA")
if _LoomCompareDNA:
    _LoomCompareDNA.restype = ctypes.c_char_p
    _LoomCompareDNA.argtypes = [ctypes.c_char_p, ctypes.c_char_p]

_LoomExtractNetworkBlueprint = _sym("LoomExtractNetworkBlueprint")
if _LoomExtractNetworkBlueprint:
    _LoomExtractNetworkBlueprint.restype = ctypes.c_char_p
    _LoomExtractNetworkBlueprint.argtypes = [ctypes.c_longlong, ctypes.c_char_p]

_LoomNewVolumetricNetwork = _sym("LoomNewVolumetricNetwork")
if _LoomNewVolumetricNetwork:
    _LoomNewVolumetricNetwork.restype = ctypes.c_longlong
    _LoomNewVolumetricNetwork.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]

# Systolic state (polymorphic execution engine)
_LoomCreateSystolicState = _sym("LoomCreateSystolicState")
if _LoomCreateSystolicState:
    _LoomCreateSystolicState.restype = ctypes.c_longlong
    _LoomCreateSystolicState.argtypes = [ctypes.c_longlong, ctypes.c_int]

_LoomFreeSystolicState = _sym("LoomFreeSystolicState")
if _LoomFreeSystolicState:
    _LoomFreeSystolicState.argtypes = [ctypes.c_longlong]

_LoomSetInput = _sym("LoomSetInput")
if _LoomSetInput:
    _LoomSetInput.restype = None
    _LoomSetInput.argtypes = [
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]

_LoomSystolicStep = _sym("LoomSystolicStep")
if _LoomSystolicStep:
    _LoomSystolicStep.restype = ctypes.c_longlong
    _LoomSystolicStep.argtypes = [ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int]

_LoomGetOutput = _sym("LoomGetOutput")
if _LoomGetOutput:
    _LoomGetOutput.restype = ctypes.c_char_p
    _LoomGetOutput.argtypes = [ctypes.c_longlong, ctypes.c_int]

_LoomSequentialForward = _sym("LoomSequentialForward")
if _LoomSequentialForward:
    _LoomSequentialForward.restype = ctypes.c_char_p
    _LoomSequentialForward.argtypes = [
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]

# Training / gradients
_LoomSystolicBackward = _sym("LoomSystolicBackward")
if _LoomSystolicBackward:
    _LoomSystolicBackward.restype = ctypes.c_char_p
    _LoomSystolicBackward.argtypes = [
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]

_LoomApplyGradients = _sym("LoomApplyGradients")
if _LoomApplyGradients:
    _LoomApplyGradients.restype = None
    _LoomApplyGradients.argtypes = [
        ctypes.c_longlong,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_float,
    ]

_LoomApplyRecursiveGradients = _sym("LoomApplyRecursiveGradients")
if _LoomApplyRecursiveGradients:
    _LoomApplyRecursiveGradients.restype = None
    _LoomApplyRecursiveGradients.argtypes = [
        ctypes.c_longlong,
        ctypes.c_char_p,
        ctypes.c_float,
    ]

_LoomComputeLossGradient = _sym("LoomComputeLossGradient")
if _LoomComputeLossGradient:
    _LoomComputeLossGradient.restype = ctypes.c_char_p
    _LoomComputeLossGradient.argtypes = [
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_char_p,
    ]

_LoomApplyTargetProp = _sym("LoomApplyTargetProp")
if _LoomApplyTargetProp:
    _LoomApplyTargetProp.restype = None
    _LoomApplyTargetProp.argtypes = [
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_float,
    ]

# TargetProp state
_LoomCreateTargetPropState = _sym("LoomCreateTargetPropState")
if _LoomCreateTargetPropState:
    _LoomCreateTargetPropState.restype = ctypes.c_longlong
    _LoomCreateTargetPropState.argtypes = [ctypes.c_longlong, ctypes.c_int]

_LoomTargetPropForward = _sym("LoomTargetPropForward")
if _LoomTargetPropForward:
    _LoomTargetPropForward.restype = ctypes.c_char_p
    _LoomTargetPropForward.argtypes = [
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]

_LoomTargetPropBackward = _sym("LoomTargetPropBackward")
if _LoomTargetPropBackward:
    _LoomTargetPropBackward.restype = ctypes.c_char_p
    _LoomTargetPropBackward.argtypes = [
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]

_LoomTargetPropBackwardChainRule = _sym("LoomTargetPropBackwardChainRule")
if _LoomTargetPropBackwardChainRule:
    _LoomTargetPropBackwardChainRule.restype = ctypes.c_char_p
    _LoomTargetPropBackwardChainRule.argtypes = [
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_float,
    ]

_LoomTargetPropBackwardTargetProp = _sym("LoomTargetPropBackwardTargetProp")
if _LoomTargetPropBackwardTargetProp:
    _LoomTargetPropBackwardTargetProp.restype = ctypes.c_char_p
    _LoomTargetPropBackwardTargetProp.argtypes = [
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_float,
    ]

_LoomDefaultTargetPropConfig = _sym("LoomDefaultTargetPropConfig")
if _LoomDefaultTargetPropConfig:
    _LoomDefaultTargetPropConfig.restype = ctypes.c_char_p
    _LoomDefaultTargetPropConfig.argtypes = []

# Weight morphing (numerical type switching)
_LoomMorphLayer = _sym("LoomMorphLayer")
if _LoomMorphLayer:
    _LoomMorphLayer.restype = ctypes.c_char_p
    _LoomMorphLayer.argtypes = [ctypes.c_longlong, ctypes.c_int, ctypes.c_int]

# WebGPU / hardware acceleration
_LoomInitWGPU = _sym("LoomInitWGPU")
if _LoomInitWGPU:
    _LoomInitWGPU.restype = ctypes.c_char_p
    _LoomInitWGPU.argtypes = [ctypes.c_longlong]

_LoomSyncToGPU = _sym("LoomSyncToGPU")
if _LoomSyncToGPU:
    _LoomSyncToGPU.restype = ctypes.c_char_p
    _LoomSyncToGPU.argtypes = [ctypes.c_longlong]

_LoomSyncToCPU = _sym("LoomSyncToCPU")
if _LoomSyncToCPU:
    _LoomSyncToCPU.restype = ctypes.c_char_p
    _LoomSyncToCPU.argtypes = [ctypes.c_longlong]

_LoomForwardWGPU = _sym("LoomForwardWGPU")
if _LoomForwardWGPU:
    _LoomForwardWGPU.restype = ctypes.c_char_p
    _LoomForwardWGPU.argtypes = [
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]

_LoomForwardTokenIDsWGPU = _sym("LoomForwardTokenIDsWGPU")
if _LoomForwardTokenIDsWGPU:
    _LoomForwardTokenIDsWGPU.restype = ctypes.c_char_p
    _LoomForwardTokenIDsWGPU.argtypes = [ctypes.c_longlong, ctypes.c_char_p]

# GPU Buffer Management
_LoomCreateGPUBuffer = _sym("LoomCreateGPUBuffer")
if _LoomCreateGPUBuffer:
    _LoomCreateGPUBuffer.restype = ctypes.c_longlong
    _LoomCreateGPUBuffer.argtypes = [ctypes.c_longlong, ctypes.c_longlong]

_LoomFreeGPUBuffer = _sym("LoomFreeGPUBuffer")
if _LoomFreeGPUBuffer:
    _LoomFreeGPUBuffer.restype = None
    _LoomFreeGPUBuffer.argtypes = [ctypes.c_longlong]

# Shader Source Getters
_LoomShaderDenseBackwardDX = _sym("LoomShaderDenseBackwardDX")
if _LoomShaderDenseBackwardDX:
    _LoomShaderDenseBackwardDX.restype = ctypes.c_char_p
    _LoomShaderDenseBackwardDX.argtypes = [ctypes.c_int]

_LoomShaderDenseBackwardDW = _sym("LoomShaderDenseBackwardDW")
if _LoomShaderDenseBackwardDW:
    _LoomShaderDenseBackwardDW.restype = ctypes.c_char_p
    _LoomShaderDenseBackwardDW.argtypes = [ctypes.c_int]

# Dense Backward
_LoomDispatchDenseBackwardDX = _sym("LoomDispatchDenseBackwardDX")
if _LoomDispatchDenseBackwardDX:
    _LoomDispatchDenseBackwardDX.restype = ctypes.c_char_p
    _LoomDispatchDenseBackwardDX.argtypes = [
        ctypes.c_longlong, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int,
    ]

_LoomDispatchDenseBackwardDW = _sym("LoomDispatchDenseBackwardDW")
if _LoomDispatchDenseBackwardDW:
    _LoomDispatchDenseBackwardDW.restype = ctypes.c_char_p
    _LoomDispatchDenseBackwardDW.argtypes = [
        ctypes.c_longlong, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int,
    ]

# SwiGLU Backward
_LoomDispatchSwiGLUBackward = _sym("LoomDispatchSwiGLUBackward")
if _LoomDispatchSwiGLUBackward:
    _LoomDispatchSwiGLUBackward.restype = ctypes.c_char_p
    _LoomDispatchSwiGLUBackward.argtypes = [
        ctypes.c_longlong, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
        ctypes.c_longlong, ctypes.c_longlong,
    ]

# RMSNorm Backward
_LoomDispatchRMSNormBackward = _sym("LoomDispatchRMSNormBackward")
if _LoomDispatchRMSNormBackward:
    _LoomDispatchRMSNormBackward.restype = ctypes.c_char_p
    _LoomDispatchRMSNormBackward.argtypes = [
        ctypes.c_longlong, ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
        ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
    ]

# Embedding Backward
_LoomDispatchEmbeddingBackward = _sym("LoomDispatchEmbeddingBackward")
if _LoomDispatchEmbeddingBackward:
    _LoomDispatchEmbeddingBackward.restype = ctypes.c_char_p
    _LoomDispatchEmbeddingBackward.argtypes = [
        ctypes.c_longlong, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
    ]

# Residual Backward
_LoomDispatchResidualBackward = _sym("LoomDispatchResidualBackward")
if _LoomDispatchResidualBackward:
    _LoomDispatchResidualBackward.restype = ctypes.c_char_p
    _LoomDispatchResidualBackward.argtypes = [
        ctypes.c_longlong, ctypes.c_int,
        ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
    ]

# CNN1 Backward
_LoomDispatchCNN1BackwardDX = _sym("LoomDispatchCNN1BackwardDX")
if _LoomDispatchCNN1BackwardDX:
    _LoomDispatchCNN1BackwardDX.restype = ctypes.c_char_p
    _LoomDispatchCNN1BackwardDX.argtypes = [
        ctypes.c_longlong, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int,
        ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
    ]

_LoomDispatchCNN1BackwardDW = _sym("LoomDispatchCNN1BackwardDW")
if _LoomDispatchCNN1BackwardDW:
    _LoomDispatchCNN1BackwardDW.restype = ctypes.c_char_p
    _LoomDispatchCNN1BackwardDW.argtypes = [
        ctypes.c_longlong, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int,
        ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
    ]

# CNN2 Backward
_LoomDispatchCNN2BackwardDX = _sym("LoomDispatchCNN2BackwardDX")
if _LoomDispatchCNN2BackwardDX:
    _LoomDispatchCNN2BackwardDX.restype = ctypes.c_char_p
    _LoomDispatchCNN2BackwardDX.argtypes = [
        ctypes.c_longlong, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int,
        ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
    ]

_LoomDispatchCNN2BackwardDW = _sym("LoomDispatchCNN2BackwardDW")
if _LoomDispatchCNN2BackwardDW:
    _LoomDispatchCNN2BackwardDW.restype = ctypes.c_char_p
    _LoomDispatchCNN2BackwardDW.argtypes = [
        ctypes.c_longlong, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int,
        ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
    ]

# CNN3 Backward
_LoomDispatchCNN3BackwardDX = _sym("LoomDispatchCNN3BackwardDX")
if _LoomDispatchCNN3BackwardDX:
    _LoomDispatchCNN3BackwardDX.restype = ctypes.c_char_p
    _LoomDispatchCNN3BackwardDX.argtypes = [
        ctypes.c_longlong, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int,
        ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
    ]

_LoomDispatchCNN3BackwardDW = _sym("LoomDispatchCNN3BackwardDW")
if _LoomDispatchCNN3BackwardDW:
    _LoomDispatchCNN3BackwardDW.restype = ctypes.c_char_p
    _LoomDispatchCNN3BackwardDW.argtypes = [
        ctypes.c_longlong, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int,
        ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
    ]

# MHA Backward
_LoomDispatchMHABackward = _sym("LoomDispatchMHABackward")
if _LoomDispatchMHABackward:
    _LoomDispatchMHABackward.restype = ctypes.c_char_p
    _LoomDispatchMHABackward.argtypes = [
        ctypes.c_longlong, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float,
        ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
        ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
    ]

# Apply Gradients
_LoomDispatchApplyGradients = _sym("LoomDispatchApplyGradients")
if _LoomDispatchApplyGradients:
    _LoomDispatchApplyGradients.restype = ctypes.c_char_p
    _LoomDispatchApplyGradients.argtypes = [
        ctypes.c_longlong, ctypes.c_int, ctypes.c_float,
        ctypes.c_longlong, ctypes.c_longlong,
    ]

# MSE Grad + Partial Loss
_LoomDispatchMSEGradPartialLoss = _sym("LoomDispatchMSEGradPartialLoss")
if _LoomDispatchMSEGradPartialLoss:
    _LoomDispatchMSEGradPartialLoss.restype = ctypes.c_char_p
    _LoomDispatchMSEGradPartialLoss.argtypes = [
        ctypes.c_longlong, ctypes.c_int,
        ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
    ]

# Layer-level Dispatch
_LoomDispatchForwardLayer = _sym("LoomDispatchForwardLayer")
if _LoomDispatchForwardLayer:
    _LoomDispatchForwardLayer.restype = ctypes.c_char_p
    _LoomDispatchForwardLayer.argtypes = [
        ctypes.c_longlong, ctypes.c_int, ctypes.c_int,
        ctypes.c_longlong, ctypes.c_longlong,
    ]

_LoomDispatchBackwardLayer = _sym("LoomDispatchBackwardLayer")
if _LoomDispatchBackwardLayer:
    _LoomDispatchBackwardLayer.restype = ctypes.c_char_p
    _LoomDispatchBackwardLayer.argtypes = [
        ctypes.c_longlong, ctypes.c_int, ctypes.c_int,
        ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
        ctypes.c_longlong, ctypes.c_longlong,
    ]

# Activation Dispatch
_LoomDispatchActivation = _sym("LoomDispatchActivation")
if _LoomDispatchActivation:
    _LoomDispatchActivation.restype = ctypes.c_char_p
    _LoomDispatchActivation.argtypes = [
        ctypes.c_longlong, ctypes.c_int, ctypes.c_int,
        ctypes.c_longlong, ctypes.c_longlong,
    ]

_LoomDispatchActivationBackward = _sym("LoomDispatchActivationBackward")
if _LoomDispatchActivationBackward:
    _LoomDispatchActivationBackward.restype = ctypes.c_char_p
    _LoomDispatchActivationBackward.argtypes = [
        ctypes.c_longlong, ctypes.c_int, ctypes.c_int,
        ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
    ]

# SafeTensors / model I/O
_LoomLoadSafetensors = _sym("LoomLoadSafetensors")
if _LoomLoadSafetensors:
    _LoomLoadSafetensors.restype = ctypes.c_char_p
    _LoomLoadSafetensors.argtypes = [ctypes.c_char_p]

_LoomLoadSafetensorsFromBytes = _sym("LoomLoadSafetensorsFromBytes")
if _LoomLoadSafetensorsFromBytes:
    _LoomLoadSafetensorsFromBytes.restype = ctypes.c_char_p
    _LoomLoadSafetensorsFromBytes.argtypes = [ctypes.c_char_p, ctypes.c_int]

_LoomLoadSafetensorsWithShapes = _sym("LoomLoadSafetensorsWithShapes")
if _LoomLoadSafetensorsWithShapes:
    _LoomLoadSafetensorsWithShapes.restype = ctypes.c_char_p
    _LoomLoadSafetensorsWithShapes.argtypes = [ctypes.c_char_p]

_LoomLoadWithPrefixes = _sym("LoomLoadWithPrefixes")
if _LoomLoadWithPrefixes:
    _LoomLoadWithPrefixes.restype = ctypes.c_char_p
    _LoomLoadWithPrefixes.argtypes = [ctypes.c_longlong, ctypes.c_char_p]

# Tokenizer
_LoomLoadTokenizer = _sym("LoomLoadTokenizer")
if _LoomLoadTokenizer:
    _LoomLoadTokenizer.restype = ctypes.c_longlong
    _LoomLoadTokenizer.argtypes = [ctypes.c_char_p]

_LoomTokenize = _sym("LoomTokenize")
if _LoomTokenize:
    _LoomTokenize.restype = ctypes.c_char_p
    _LoomTokenize.argtypes = [ctypes.c_longlong, ctypes.c_char_p]

_LoomDetokenize = _sym("LoomDetokenize")
if _LoomDetokenize:
    _LoomDetokenize.restype = ctypes.c_char_p
    _LoomDetokenize.argtypes = [ctypes.c_longlong, ctypes.c_char_p]

_LoomFreeTokenizer = _sym("LoomFreeTokenizer")
if _LoomFreeTokenizer:
    _LoomFreeTokenizer.argtypes = [ctypes.c_longlong]

# Per-layer forward dispatch (for granular control)
_LAYER_NAMES = [
    "Dense", "RMSNorm", "LayerNorm", "MHA", "Softmax", "SwiGLU",
    "Embedding", "Residual", "KMeans", "RNN", "LSTM",
    "CNN1", "CNN2", "CNN3",
    "ConvTransposed1D", "ConvTransposed2D", "ConvTransposed3D",
]

_layer_forward_fns: Dict[str, Any] = {}
for _ln in _LAYER_NAMES:
    _fn = _sym(f"Loom{_ln}Forward")
    if _fn:
        _fn.restype = ctypes.c_char_p
        _fn.argtypes = [ctypes.c_longlong, ctypes.c_int, ctypes.c_longlong]
        _layer_forward_fns[_ln] = _fn

_layer_backward_fns: Dict[str, Any] = {}
for _ln in _LAYER_NAMES:
    _fn = _sym(f"Loom{_ln}Backward")
    if _fn:
        _fn.restype = ctypes.c_char_p
        _fn.argtypes = [ctypes.c_longlong, ctypes.c_int, ctypes.c_longlong]
        _layer_backward_fns[_ln] = _fn


# ---------------------------------------------------------------------------
# Type Definitions
# ---------------------------------------------------------------------------

class DType:
    """
    Numerical data types supported by the M-POLY-VTD engine.

    Each layer's WeightStore can hold any of these 21 types simultaneously,
    and can morph between them instantly without re-allocation.
    """
    FLOAT64   = 0   # 64-bit double
    FLOAT32   = 1   # Standard 32-bit float (default / source of truth)
    FLOAT16   = 2   # 16-bit half precision
    BFLOAT16  = 3   # Brain Float 16 (better dynamic range than FP16)
    FP8_E4M3  = 4   # 8-bit FP8 (activations / weights)
    FP8_E5M2  = 5   # 8-bit FP8 (gradients)
    INT64     = 6
    INT32     = 7
    INT16     = 8
    INT8      = 9
    UINT64    = 10
    UINT32    = 11
    UINT16    = 12
    UINT8     = 13
    INT4      = 14
    UINT4     = 15
    FP4       = 16  # 4-bit E2M1 — extreme compression
    INT2      = 17
    UINT2     = 18
    TERNARY   = 19  # {-1, 0, 1}
    BINARY    = 20  # 1-bit XNOR-Net

    _NAMES = {
        0: "float64", 1: "float32", 2: "float16", 3: "bfloat16",
        4: "fp8_e4m3", 5: "fp8_e5m2",
        6: "int64", 7: "int32", 8: "int16", 9: "int8",
        10: "uint64", 11: "uint32", 12: "uint16", 13: "uint8",
        14: "int4", 15: "uint4", 16: "fp4",
        17: "int2", 18: "uint2", 19: "ternary", 20: "binary",
    }

    @classmethod
    def name(cls, dtype: int) -> str:
        return cls._NAMES.get(dtype, f"dtype({dtype})")


class LayerType:
    """Layer types available in the volumetric network grid."""
    DENSE           = 0
    MHA             = 1   # Multi-Head Attention
    SWIGLU          = 2   # Swish-Gated Linear Unit (LLaMA activation)
    RMS_NORM        = 3
    CNN1            = 4   # 1D Convolution
    CNN2            = 5   # 2D Convolution
    CNN3            = 6   # 3D Convolution
    CONV_TRANS_1D   = 7
    CONV_TRANS_2D   = 8
    CONV_TRANS_3D   = 9
    RNN             = 10
    LSTM            = 11
    LAYER_NORM      = 12
    EMBEDDING       = 13
    KMEANS          = 14  # Unsupervised clustering layer
    SOFTMAX         = 15
    PARALLEL        = 16  # MoE / Ensemble
    SEQUENTIAL      = 17
    RESIDUAL        = 18


class Activation:
    """
    Activation functions for layer configurations.

    Note: SiLU and GELU are new in M-POLY-VTD replacing the old
    ScaledReLU and Softplus variants.
    """
    RELU    = 0
    SILU    = 1   # Swish / SiLU — preferred for modern architectures
    GELU    = 2   # Gaussian Error Linear Unit
    TANH    = 3
    SIGMOID = 4
    LINEAR  = 5

    # Legacy aliases kept for backwards compatibility
    SCALED_RELU = 0
    LEAKY_RELU  = 0
    SOFTPLUS    = 5


# ---------------------------------------------------------------------------
# Low-level functional API
# ---------------------------------------------------------------------------

def build_network(json_config) -> int:
    """
    Build a VolumetricNetwork from a JSON configuration.

    Args:
        json_config: JSON string or dict describing the network topology.
                     Use poly.BuildNetworkFromJSON schema.

    Returns:
        Network handle (integer). Free with free_network() when done.

    Example::

        config = {
            "id": "my_net",
            "depth": 1, "rows": 1, "cols": 3,
            "layers": [
                {"type": "dense", "input_size": 128, "output_size": 256,
                 "activation": "silu", "dtype": "float32"},
                {"type": "dense", "input_size": 256, "output_size": 128,
                 "activation": "silu", "dtype": "float32"},
                {"type": "dense", "input_size": 128, "output_size": 10,
                 "activation": "sigmoid", "dtype": "float32"},
            ]
        }
        net = build_network(config)
    """
    if not _LoomBuildNetworkFromJSON:
        raise RuntimeError("LoomBuildNetworkFromJSON not available in library")
    if isinstance(json_config, dict):
        json_config = json.dumps(json_config)
    handle = _LoomBuildNetworkFromJSON(json_config.encode("utf-8"))
    if handle < 0:
        raise RuntimeError("Failed to build network (invalid config or allocation error)")
    return int(handle)


def load_network(path: str) -> int:
    """
    Load a network from a file (SafeTensors or universal format).

    Args:
        path: File path.

    Returns:
        Network handle. Free with free_network().
    """
    if not _LoomLoadUniversal:
        raise RuntimeError("LoomLoadUniversal not available in library")
    handle = _LoomLoadUniversal(path.encode("utf-8"))
    if handle < 0:
        raise RuntimeError(f"Failed to load network from {path!r}")
    return int(handle)


def load_network_detailed(path: str) -> dict:
    """
    Load a network from file and return handle + metadata.

    Returns:
        Dict with 'handle', 'layers', 'grid' keys.
    """
    if not _LoomLoadUniversalDetailed:
        raise RuntimeError("LoomLoadUniversalDetailed not available in library")
    result = _parse_json(_LoomLoadUniversalDetailed(path.encode("utf-8")))
    return _check(result, "load_network_detailed")


def free_network(handle: int) -> None:
    """Free all resources associated with a network handle."""
    if _LoomFreeNetwork and handle >= 0:
        _LoomFreeNetwork(int(handle))


def get_network_info(handle: int) -> dict:
    """Return metadata about the network (total_layers, grid dimensions)."""
    if not _LoomGetNetworkInfo:
        raise RuntimeError("LoomGetNetworkInfo not available in library")
    result = _parse_json(_LoomGetNetworkInfo(int(handle)))
    return _check(result, "get_network_info")


def get_layer_telemetry(handle: int, layer_idx: int) -> dict:
    """
    Return telemetry (weights stats, dtype, shape) for a specific layer.

    Args:
        handle: Network handle.
        layer_idx: Zero-based layer index in n.Layers.
    """
    if not _LoomGetLayerTelemetry:
        raise RuntimeError("LoomGetLayerTelemetry not available in library")
    result = _parse_json(_LoomGetLayerTelemetry(int(handle), int(layer_idx)))
    return _check(result, f"get_layer_telemetry(layer={layer_idx})")


def morph_layer(handle: int, layer_idx: int, target_dtype: int) -> dict:
    """
    Morph a layer's numerical type at runtime (zero-cost when cached).

    This is the core M-POLY-VTD operation — switch any layer between any of
    the 21 supported DType values without re-allocating the weight store.

    Args:
        handle: Network handle.
        layer_idx: Layer index.
        target_dtype: Target DType constant (e.g. DType.INT8).

    Returns:
        Status dict.

    Example::

        morph_layer(net, 0, DType.INT8)   # quantize layer 0 to INT8
        morph_layer(net, 1, DType.FP4)    # extreme compression on layer 1
    """
    if not _LoomMorphLayer:
        raise RuntimeError("LoomMorphLayer not available in library")
    result = _parse_json(_LoomMorphLayer(int(handle), int(layer_idx), int(target_dtype)))
    return _check(result, f"morph_layer(layer={layer_idx}, dtype={DType.name(target_dtype)})")


def get_available_methods() -> List[str]:
    """Return the list of C-ABI methods exported by the loaded library."""
    if not _LoomGetMethodsJSON:
        return []
    result = _parse_json(_LoomGetMethodsJSON())
    return result if isinstance(result, list) else []


# ---------------------------------------------------------------------------
# SystolicState — polymorphic forward/backward execution
# ---------------------------------------------------------------------------

def create_systolic_state(network_handle: int, dtype: int = DType.FLOAT32) -> int:
    """
    Create a typed execution state for the network.

    The SystolicState manages per-layer activations and is the vehicle for
    all forward and backward passes. Create one per inference/training call.

    Args:
        network_handle: Handle from build_network().
        dtype: Numerical type to use for this execution (default: FLOAT32).
               Inputs are always accepted as float32 and cast internally.

    Returns:
        State handle. Free with free_systolic_state().
    """
    if not _LoomCreateSystolicState:
        raise RuntimeError("LoomCreateSystolicState not available in library")
    handle = _LoomCreateSystolicState(int(network_handle), int(dtype))
    if handle < 0:
        raise RuntimeError("Failed to create SystolicState")
    return int(handle)


def free_systolic_state(handle: int) -> None:
    """Free a SystolicState handle."""
    if _LoomFreeSystolicState and handle >= 0:
        _LoomFreeSystolicState(int(handle))


def set_input(state_handle: int, data: List[float]) -> None:
    """
    Load input data into a SystolicState before calling systolic_step().

    Args:
        state_handle: Handle from create_systolic_state().
        data: Input vector (float values — cast to state dtype internally).
    """
    if not _LoomSetInput:
        raise RuntimeError("LoomSetInput not available in library")
    arr, length = _to_cfloat_array(data)
    _LoomSetInput(int(state_handle), arr, length)


def systolic_step(network_handle: int, state_handle: int,
                  capture_history: bool = False) -> int:
    """
    Execute one full forward pass through the volumetric network.

    Args:
        network_handle: Network handle.
        state_handle: SystolicState handle (must have input set via set_input()).
        capture_history: If True, stores per-layer activations for backward pass.

    Returns:
        Execution duration in nanoseconds.
    """
    if not _LoomSystolicStep:
        raise RuntimeError("LoomSystolicStep not available in library")
    return int(_LoomSystolicStep(
        int(network_handle), int(state_handle), int(capture_history)
    ))


def get_output(state_handle: int, layer_idx: int = -1) -> List[float]:
    """
    Retrieve the output tensor from a layer after systolic_step().

    Args:
        state_handle: SystolicState handle.
        layer_idx: Layer index to read (use -1 to read the last layer via
                   layer_count - 1 — caller must supply the correct index).

    Returns:
        Output values as a list of floats.
    """
    if not _LoomGetOutput:
        raise RuntimeError("LoomGetOutput not available in library")
    result = _parse_json(_LoomGetOutput(int(state_handle), int(layer_idx)))
    if result is None:
        raise RuntimeError("No output available — did you call systolic_step()?")
    _check(result, "get_output")
    return result if isinstance(result, list) else []


def sequential_forward(network_handle: int, inputs: List[float]) -> List[float]:
    """
    One-shot forward pass — no state management required.

    Internally creates a transient SystolicState, runs the forward pass,
    extracts the final layer output, and disposes of the state. Uses the
    dtype of the first layer in the network.

    Args:
        network_handle: Network handle.
        inputs: Input vector.

    Returns:
        Output vector (float32).
    """
    if not _LoomSequentialForward:
        raise RuntimeError("LoomSequentialForward not available in library")
    arr, length = _to_cfloat_array(inputs)
    result = _parse_json(_LoomSequentialForward(int(network_handle), arr, length))
    _check(result, "sequential_forward")
    return result if isinstance(result, list) else []


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def systolic_backward(network_handle: int, state_handle: int,
                      grad_output: List[float]) -> List[float]:
    """
    Run backpropagation through the network.

    Must be called after systolic_step() with capture_history=True.

    Args:
        network_handle: Network handle.
        state_handle: SystolicState handle (with captured history).
        grad_output: Loss gradient w.r.t. the network output.

    Returns:
        Gradient w.r.t. the network input.
    """
    if not _LoomSystolicBackward:
        raise RuntimeError("LoomSystolicBackward not available in library")
    arr, length = _to_cfloat_array(grad_output)
    result = _parse_json(
        _LoomSystolicBackward(int(network_handle), int(state_handle), arr, length)
    )
    _check(result, "systolic_backward")
    return result if isinstance(result, list) else []


def compute_loss_gradient(state_handle: int, targets: List[float],
                          loss_type: str = "mse") -> List[float]:
    """
    Compute the loss gradient between the network output and target values.

    Args:
        state_handle: SystolicState handle (after forward pass).
        targets: Target output values.
        loss_type: Loss function: "mse", "cross_entropy", "mae", etc.

    Returns:
        Gradient vector to pass to systolic_backward().
    """
    if not _LoomComputeLossGradient:
        raise RuntimeError("LoomComputeLossGradient not available in library")
    arr, length = _to_cfloat_array(targets)
    result = _parse_json(
        _LoomComputeLossGradient(
            int(state_handle), arr, length, loss_type.encode("utf-8")
        )
    )
    _check(result, "compute_loss_gradient")
    return result if isinstance(result, list) else []


def apply_gradients(network_handle: int, layer_idx: int,
                    grad_weights: List[float], learning_rate: float) -> None:
    """
    Apply computed weight gradients to a specific layer.

    Args:
        network_handle: Network handle.
        layer_idx: Layer to update.
        grad_weights: Gradient tensor as a flat float list.
        learning_rate: Step size.
    """
    if not _LoomApplyGradients:
        raise RuntimeError("LoomApplyGradients not available in library")
    grad_json = json.dumps({"Data": grad_weights}).encode("utf-8")
    _LoomApplyGradients(int(network_handle), int(layer_idx), grad_json,
                        float(learning_rate))


def apply_recursive_gradients(network_handle: int, grad_json: dict,
                               learning_rate: float) -> None:
    """
    Apply gradients to all layers at once using a pre-computed gradient map.

    Args:
        network_handle: Network handle.
        grad_json: Dict mapping layer indices to gradient tensors.
        learning_rate: Step size.
    """
    if not _LoomApplyRecursiveGradients:
        raise RuntimeError("LoomApplyRecursiveGradients not available in library")
    _LoomApplyRecursiveGradients(
        int(network_handle),
        json.dumps(grad_json).encode("utf-8"),
        float(learning_rate),
    )


def apply_target_prop(network_handle: int, state_handle: int,
                      target: List[float], learning_rate: float) -> None:
    """
    Apply Neural Target Propagation — a backprop-free local learning rule.

    Target Propagation uses gap-based updates: delta = lr * input * gap
    instead of chain rule gradients, enabling learning in non-differentiable
    or discrete networks.

    Args:
        network_handle: Network handle.
        state_handle: SystolicState handle (after forward pass).
        target: Target output values.
        learning_rate: Step size.
    """
    if not _LoomApplyTargetProp:
        raise RuntimeError("LoomApplyTargetProp not available in library")
    arr, length = _to_cfloat_array(target)
    _LoomApplyTargetProp(
        int(network_handle), int(state_handle), arr, length, float(learning_rate)
    )


# ---------------------------------------------------------------------------
# TargetPropState — explicit target propagation session
# ---------------------------------------------------------------------------

def create_target_prop_state(network_handle: int,
                              dtype: int = DType.FLOAT32) -> int:
    """
    Create an explicit TargetPropState for fine-grained target propagation.

    Returns:
        TargetPropState handle.
    """
    if not _LoomCreateTargetPropState:
        raise RuntimeError("LoomCreateTargetPropState not available in library")
    handle = _LoomCreateTargetPropState(int(network_handle), int(dtype))
    if handle < 0:
        raise RuntimeError("Failed to create TargetPropState")
    return int(handle)


def target_prop_forward(network_handle: int, tp_handle: int,
                        inputs: List[float]) -> List[float]:
    """Run a forward pass using the TargetPropState."""
    if not _LoomTargetPropForward:
        raise RuntimeError("LoomTargetPropForward not available in library")
    arr, length = _to_cfloat_array(inputs)
    result = _parse_json(
        _LoomTargetPropForward(int(network_handle), int(tp_handle), arr, length)
    )
    _check(result, "target_prop_forward")
    return result if isinstance(result, list) else []


def target_prop_backward(network_handle: int, tp_handle: int,
                         target: List[float]) -> List[float]:
    """Run a backward pass using standard target propagation."""
    if not _LoomTargetPropBackward:
        raise RuntimeError("LoomTargetPropBackward not available in library")
    arr, length = _to_cfloat_array(target)
    result = _parse_json(
        _LoomTargetPropBackward(int(network_handle), int(tp_handle), arr, length)
    )
    _check(result, "target_prop_backward")
    return result if isinstance(result, list) else []


def target_prop_backward_chain_rule(network_handle: int, tp_handle: int,
                                    target: List[float],
                                    learning_rate: float) -> List[float]:
    """Run target prop backward with chain-rule hybrid."""
    if not _LoomTargetPropBackwardChainRule:
        raise RuntimeError("LoomTargetPropBackwardChainRule not available in library")
    arr, length = _to_cfloat_array(target)
    result = _parse_json(
        _LoomTargetPropBackwardChainRule(
            int(network_handle), int(tp_handle), arr, length, float(learning_rate)
        )
    )
    _check(result, "target_prop_backward_chain_rule")
    return result if isinstance(result, list) else []


def get_default_target_prop_config() -> dict:
    """Return the default TargetPropConfig as a dict."""
    if not _LoomDefaultTargetPropConfig:
        return {}
    result = _parse_json(_LoomDefaultTargetPropConfig())
    return result or {}


# ---------------------------------------------------------------------------
# GPU / WebGPU acceleration
# ---------------------------------------------------------------------------

def init_wgpu(network_handle: int) -> dict:
    """
    Initialize WebGPU for a network. Must be called before forward_wgpu().

    Returns:
        Status dict with 'status' or 'error'.
    """
    if not _LoomInitWGPU:
        raise RuntimeError("LoomInitWGPU not available in library")
    result = _parse_json(_LoomInitWGPU(int(network_handle)))
    return _check(result, "init_wgpu")


def sync_to_gpu(network_handle: int) -> dict:
    """Upload all weight tensors to GPU VRAM."""
    if not _LoomSyncToGPU:
        raise RuntimeError("LoomSyncToGPU not available in library")
    result = _parse_json(_LoomSyncToGPU(int(network_handle)))
    return _check(result, "sync_to_gpu")


def sync_to_cpu(network_handle: int) -> dict:
    """Download all weight tensors from GPU back to CPU."""
    if not _LoomSyncToCPU:
        raise RuntimeError("LoomSyncToCPU not available in library")
    result = _parse_json(_LoomSyncToCPU(int(network_handle)))
    return _check(result, "sync_to_cpu")


def forward_wgpu(network_handle: int, inputs: List[float]) -> List[float]:
    """
    Run a GPU-accelerated forward pass.

    Requires init_wgpu() and sync_to_gpu() to have been called first.

    Args:
        network_handle: Network handle.
        inputs: Input vector.

    Returns:
        Output vector (float32).
    """
    if not _LoomForwardWGPU:
        raise RuntimeError("LoomForwardWGPU not available in library")
    arr, length = _to_cfloat_array(inputs)
    result = _parse_json(_LoomForwardWGPU(int(network_handle), arr, length))
    _check(result, "forward_wgpu")
    return result if isinstance(result, list) else []


def forward_token_ids_wgpu(network_handle: int,
                            token_ids: List[int]) -> List[float]:
    """
    GPU-accelerated forward pass from integer token IDs (for LLM inference).

    Args:
        network_handle: Network handle (must have Embedding layer).
        token_ids: List of integer token IDs.

    Returns:
        Output logits as float list.
    """
    if not _LoomForwardTokenIDsWGPU:
        raise RuntimeError("LoomForwardTokenIDsWGPU not available in library")
    ids_json = json.dumps(token_ids).encode("utf-8")
    result = _parse_json(_LoomForwardTokenIDsWGPU(int(network_handle), ids_json))
    _check(result, "forward_token_ids_wgpu")
    return result if isinstance(result, list) else []


# ---------------------------------------------------------------------------
# GPU Buffer Management
# ---------------------------------------------------------------------------

def create_gpu_buffer(network_handle: int, size_bytes: int) -> int:
    """
    Allocate a GPU-resident storage buffer on the device.

    Args:
        network_handle: Network handle (must have WGPU initialized).
        size_bytes: Buffer size in bytes.

    Returns:
        Buffer handle (int64) for use with dispatch functions.
    """
    if not _LoomCreateGPUBuffer:
        raise RuntimeError("LoomCreateGPUBuffer not available in library")
    handle = _LoomCreateGPUBuffer(int(network_handle), int(size_bytes))
    if handle < 0:
        raise RuntimeError("Failed to create GPU buffer (WGPU not initialized?)")
    return int(handle)


def free_gpu_buffer(buf_handle: int) -> None:
    """Destroy a GPU buffer and release its handle."""
    if not _LoomFreeGPUBuffer:
        raise RuntimeError("LoomFreeGPUBuffer not available in library")
    _LoomFreeGPUBuffer(int(buf_handle))


def shader_dense_backward_dx(tile_size: int = 16) -> str:
    """Return the WGSL shader source for dense backward input gradient (DX)."""
    if not _LoomShaderDenseBackwardDX:
        raise RuntimeError("LoomShaderDenseBackwardDX not available in library")
    raw = _LoomShaderDenseBackwardDX(int(tile_size))
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def shader_dense_backward_dw(tile_size: int = 16) -> str:
    """Return the WGSL shader source for dense backward weight gradient (DW)."""
    if not _LoomShaderDenseBackwardDW:
        raise RuntimeError("LoomShaderDenseBackwardDW not available in library")
    raw = _LoomShaderDenseBackwardDW(int(tile_size))
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def dispatch_dense_backward_dx(network_handle: int, batch_size: int, input_size: int,
                                output_size: int, grad_output_handle: int,
                                weight_handle: int, grad_input_handle: int,
                                tile_size: int = 16) -> dict:
    """GPU backward pass: compute input gradient for a dense layer."""
    if not _LoomDispatchDenseBackwardDX:
        raise RuntimeError("LoomDispatchDenseBackwardDX not available in library")
    result = _parse_json(_LoomDispatchDenseBackwardDX(
        int(network_handle), int(batch_size), int(input_size), int(output_size),
        int(grad_output_handle), int(weight_handle), int(grad_input_handle), int(tile_size),
    ))
    return _check(result, "dispatch_dense_backward_dx")


def dispatch_dense_backward_dw(network_handle: int, batch_size: int, input_size: int,
                                output_size: int, grad_output_handle: int,
                                input_handle: int, grad_weight_handle: int,
                                tile_size: int = 16) -> dict:
    """GPU backward pass: compute weight gradient for a dense layer."""
    if not _LoomDispatchDenseBackwardDW:
        raise RuntimeError("LoomDispatchDenseBackwardDW not available in library")
    result = _parse_json(_LoomDispatchDenseBackwardDW(
        int(network_handle), int(batch_size), int(input_size), int(output_size),
        int(grad_output_handle), int(input_handle), int(grad_weight_handle), int(tile_size),
    ))
    return _check(result, "dispatch_dense_backward_dw")


def dispatch_swiglu_backward(network_handle: int, batch_size: int, input_size: int,
                              output_size: int, grad_output_handle: int,
                              gate_in_handle: int, up_in_handle: int,
                              grad_gate_handle: int, grad_up_handle: int) -> dict:
    """GPU backward pass for SwiGLU activation."""
    if not _LoomDispatchSwiGLUBackward:
        raise RuntimeError("LoomDispatchSwiGLUBackward not available in library")
    result = _parse_json(_LoomDispatchSwiGLUBackward(
        int(network_handle), int(batch_size), int(input_size), int(output_size),
        int(grad_output_handle), int(gate_in_handle), int(up_in_handle),
        int(grad_gate_handle), int(grad_up_handle),
    ))
    return _check(result, "dispatch_swiglu_backward")


def dispatch_rmsnorm_backward(network_handle: int, batch_size: int, size: int,
                               epsilon: float, grad_output_handle: int, input_handle: int,
                               rms_handle: int, weight_handle: int,
                               grad_input_handle: int, grad_weight_handle: int) -> dict:
    """GPU backward pass for RMSNorm."""
    if not _LoomDispatchRMSNormBackward:
        raise RuntimeError("LoomDispatchRMSNormBackward not available in library")
    result = _parse_json(_LoomDispatchRMSNormBackward(
        int(network_handle), int(batch_size), int(size), float(epsilon),
        int(grad_output_handle), int(input_handle), int(rms_handle), int(weight_handle),
        int(grad_input_handle), int(grad_weight_handle),
    ))
    return _check(result, "dispatch_rmsnorm_backward")


def dispatch_embedding_backward(network_handle: int, vocab_size: int, hidden_size: int,
                                 num_tokens: int, indices_handle: int,
                                 grad_output_handle: int, grad_weight_handle: int) -> dict:
    """GPU backward pass for embedding lookup."""
    if not _LoomDispatchEmbeddingBackward:
        raise RuntimeError("LoomDispatchEmbeddingBackward not available in library")
    result = _parse_json(_LoomDispatchEmbeddingBackward(
        int(network_handle), int(vocab_size), int(hidden_size), int(num_tokens),
        int(indices_handle), int(grad_output_handle), int(grad_weight_handle),
    ))
    return _check(result, "dispatch_embedding_backward")


def dispatch_residual_backward(network_handle: int, size: int, grad_output_handle: int,
                                grad_input_handle: int, grad_residual_handle: int) -> dict:
    """GPU backward pass for residual (skip) connection."""
    if not _LoomDispatchResidualBackward:
        raise RuntimeError("LoomDispatchResidualBackward not available in library")
    result = _parse_json(_LoomDispatchResidualBackward(
        int(network_handle), int(size),
        int(grad_output_handle), int(grad_input_handle), int(grad_residual_handle),
    ))
    return _check(result, "dispatch_residual_backward")


def dispatch_cnn1_backward_dx(network_handle: int, batch_size: int, in_c: int, in_l: int,
                               filters: int, out_l: int, k_size: int, stride: int,
                               padding: int, activation: int, grad_output_handle: int,
                               weight_handle: int, pre_act_handle: int,
                               grad_input_handle: int) -> dict:
    """GPU backward DX for 1D convolution."""
    if not _LoomDispatchCNN1BackwardDX:
        raise RuntimeError("LoomDispatchCNN1BackwardDX not available in library")
    result = _parse_json(_LoomDispatchCNN1BackwardDX(
        int(network_handle), int(batch_size), int(in_c), int(in_l),
        int(filters), int(out_l), int(k_size), int(stride), int(padding), int(activation),
        int(grad_output_handle), int(weight_handle), int(pre_act_handle), int(grad_input_handle),
    ))
    return _check(result, "dispatch_cnn1_backward_dx")


def dispatch_cnn1_backward_dw(network_handle: int, batch_size: int, in_c: int, in_l: int,
                               filters: int, out_l: int, k_size: int, stride: int,
                               padding: int, activation: int, grad_output_handle: int,
                               input_handle: int, pre_act_handle: int,
                               grad_weight_handle: int) -> dict:
    """GPU backward DW for 1D convolution."""
    if not _LoomDispatchCNN1BackwardDW:
        raise RuntimeError("LoomDispatchCNN1BackwardDW not available in library")
    result = _parse_json(_LoomDispatchCNN1BackwardDW(
        int(network_handle), int(batch_size), int(in_c), int(in_l),
        int(filters), int(out_l), int(k_size), int(stride), int(padding), int(activation),
        int(grad_output_handle), int(input_handle), int(pre_act_handle), int(grad_weight_handle),
    ))
    return _check(result, "dispatch_cnn1_backward_dw")


def dispatch_cnn2_backward_dx(network_handle: int, batch_size: int, in_c: int, in_h: int,
                               in_w: int, filters: int, out_h: int, out_w: int,
                               k_size: int, stride: int, padding: int, activation: int,
                               grad_output_handle: int, weight_handle: int,
                               pre_act_handle: int, grad_input_handle: int) -> dict:
    """GPU backward DX for 2D convolution."""
    if not _LoomDispatchCNN2BackwardDX:
        raise RuntimeError("LoomDispatchCNN2BackwardDX not available in library")
    result = _parse_json(_LoomDispatchCNN2BackwardDX(
        int(network_handle), int(batch_size), int(in_c), int(in_h), int(in_w),
        int(filters), int(out_h), int(out_w), int(k_size), int(stride), int(padding), int(activation),
        int(grad_output_handle), int(weight_handle), int(pre_act_handle), int(grad_input_handle),
    ))
    return _check(result, "dispatch_cnn2_backward_dx")


def dispatch_cnn2_backward_dw(network_handle: int, batch_size: int, in_c: int, in_h: int,
                               in_w: int, filters: int, out_h: int, out_w: int,
                               k_size: int, stride: int, padding: int, activation: int,
                               grad_output_handle: int, input_handle: int,
                               pre_act_handle: int, grad_weight_handle: int) -> dict:
    """GPU backward DW for 2D convolution."""
    if not _LoomDispatchCNN2BackwardDW:
        raise RuntimeError("LoomDispatchCNN2BackwardDW not available in library")
    result = _parse_json(_LoomDispatchCNN2BackwardDW(
        int(network_handle), int(batch_size), int(in_c), int(in_h), int(in_w),
        int(filters), int(out_h), int(out_w), int(k_size), int(stride), int(padding), int(activation),
        int(grad_output_handle), int(input_handle), int(pre_act_handle), int(grad_weight_handle),
    ))
    return _check(result, "dispatch_cnn2_backward_dw")


def dispatch_cnn3_backward_dx(network_handle: int, batch_size: int, in_c: int, in_d: int,
                               in_h: int, in_w: int, filters: int, out_d: int, out_h: int,
                               out_w: int, k_size: int, stride: int, padding: int,
                               activation: int, grad_output_handle: int, weight_handle: int,
                               pre_act_handle: int, grad_input_handle: int) -> dict:
    """GPU backward DX for 3D convolution."""
    if not _LoomDispatchCNN3BackwardDX:
        raise RuntimeError("LoomDispatchCNN3BackwardDX not available in library")
    result = _parse_json(_LoomDispatchCNN3BackwardDX(
        int(network_handle), int(batch_size), int(in_c), int(in_d), int(in_h), int(in_w),
        int(filters), int(out_d), int(out_h), int(out_w), int(k_size), int(stride), int(padding), int(activation),
        int(grad_output_handle), int(weight_handle), int(pre_act_handle), int(grad_input_handle),
    ))
    return _check(result, "dispatch_cnn3_backward_dx")


def dispatch_cnn3_backward_dw(network_handle: int, batch_size: int, in_c: int, in_d: int,
                               in_h: int, in_w: int, filters: int, out_d: int, out_h: int,
                               out_w: int, k_size: int, stride: int, padding: int,
                               activation: int, grad_output_handle: int, input_handle: int,
                               pre_act_handle: int, grad_weight_handle: int) -> dict:
    """GPU backward DW for 3D convolution."""
    if not _LoomDispatchCNN3BackwardDW:
        raise RuntimeError("LoomDispatchCNN3BackwardDW not available in library")
    result = _parse_json(_LoomDispatchCNN3BackwardDW(
        int(network_handle), int(batch_size), int(in_c), int(in_d), int(in_h), int(in_w),
        int(filters), int(out_d), int(out_h), int(out_w), int(k_size), int(stride), int(padding), int(activation),
        int(grad_output_handle), int(input_handle), int(pre_act_handle), int(grad_weight_handle),
    ))
    return _check(result, "dispatch_cnn3_backward_dw")


def dispatch_mha_backward(network_handle: int, batch_size: int, num_heads: int,
                           num_kv_heads: int, head_dim: int, seq_len: int, scale: float,
                           grad_output_handle: int, q_handle: int, k_handle: int,
                           v_handle: int, dq_handle: int, dk_handle: int,
                           dv_handle: int) -> dict:
    """GPU backward pass for Multi-Head Attention."""
    if not _LoomDispatchMHABackward:
        raise RuntimeError("LoomDispatchMHABackward not available in library")
    result = _parse_json(_LoomDispatchMHABackward(
        int(network_handle), int(batch_size), int(num_heads), int(num_kv_heads),
        int(head_dim), int(seq_len), float(scale),
        int(grad_output_handle), int(q_handle), int(k_handle), int(v_handle),
        int(dq_handle), int(dk_handle), int(dv_handle),
    ))
    return _check(result, "dispatch_mha_backward")


def dispatch_apply_gradients(network_handle: int, size: int, lr: float,
                              weight_handle: int, grad_handle: int) -> dict:
    """Apply gradients to weights in-place on GPU: weight -= lr * grad."""
    if not _LoomDispatchApplyGradients:
        raise RuntimeError("LoomDispatchApplyGradients not available in library")
    result = _parse_json(_LoomDispatchApplyGradients(
        int(network_handle), int(size), float(lr),
        int(weight_handle), int(grad_handle),
    ))
    return _check(result, "dispatch_apply_gradients")


def dispatch_mse_grad_partial_loss(network_handle: int, size: int, output_handle: int,
                                    target_handle: int, grad_handle: int,
                                    partials_handle: int) -> dict:
    """
    GPU MSE loss + gradient computation.

    Computes grad = 2*(output - target)/size and writes ceil(size/256) partial
    loss sums to partials_handle. The caller sums partials on CPU for total loss.
    """
    if not _LoomDispatchMSEGradPartialLoss:
        raise RuntimeError("LoomDispatchMSEGradPartialLoss not available in library")
    result = _parse_json(_LoomDispatchMSEGradPartialLoss(
        int(network_handle), int(size),
        int(output_handle), int(target_handle), int(grad_handle), int(partials_handle),
    ))
    return _check(result, "dispatch_mse_grad_partial_loss")


def dispatch_forward_layer(network_handle: int, layer_idx: int, batch_size: int,
                            input_handle: int, output_handle: int) -> dict:
    """GPU forward pass for a single layer by index."""
    if not _LoomDispatchForwardLayer:
        raise RuntimeError("LoomDispatchForwardLayer not available in library")
    result = _parse_json(_LoomDispatchForwardLayer(
        int(network_handle), int(layer_idx), int(batch_size),
        int(input_handle), int(output_handle),
    ))
    return _check(result, "dispatch_forward_layer")


def dispatch_backward_layer(network_handle: int, layer_idx: int, batch_size: int,
                             grad_out_handle: int, input_handle: int, pre_act_handle: int,
                             dx_handle: int, dw_handle: int) -> dict:
    """GPU backward pass for a single layer by index (computes DX and DW)."""
    if not _LoomDispatchBackwardLayer:
        raise RuntimeError("LoomDispatchBackwardLayer not available in library")
    result = _parse_json(_LoomDispatchBackwardLayer(
        int(network_handle), int(layer_idx), int(batch_size),
        int(grad_out_handle), int(input_handle), int(pre_act_handle),
        int(dx_handle), int(dw_handle),
    ))
    return _check(result, "dispatch_backward_layer")


def dispatch_activation(network_handle: int, size: int, activation: int,
                         input_handle: int, output_handle: int) -> dict:
    """Apply an activation function to a GPU buffer in-place."""
    if not _LoomDispatchActivation:
        raise RuntimeError("LoomDispatchActivation not available in library")
    result = _parse_json(_LoomDispatchActivation(
        int(network_handle), int(size), int(activation),
        int(input_handle), int(output_handle),
    ))
    return _check(result, "dispatch_activation")


def dispatch_activation_backward(network_handle: int, size: int, activation: int,
                                  grad_out_handle: int, pre_act_handle: int,
                                  grad_in_handle: int) -> dict:
    """Compute activation backward gradient on GPU."""
    if not _LoomDispatchActivationBackward:
        raise RuntimeError("LoomDispatchActivationBackward not available in library")
    result = _parse_json(_LoomDispatchActivationBackward(
        int(network_handle), int(size), int(activation),
        int(grad_out_handle), int(pre_act_handle), int(grad_in_handle),
    ))
    return _check(result, "dispatch_activation_backward")


# ---------------------------------------------------------------------------
# SafeTensors / Model I/O
# ---------------------------------------------------------------------------

def load_safetensors(path: str) -> dict:
    """
    Load a SafeTensors file and return its tensors as a dict.

    Args:
        path: Path to a .safetensors file.

    Returns:
        Dict mapping tensor names to their data.
    """
    if not _LoomLoadSafetensors:
        raise RuntimeError("LoomLoadSafetensors not available in library")
    result = _parse_json(_LoomLoadSafetensors(path.encode("utf-8")))
    return _check(result, f"load_safetensors({path!r})")


def load_safetensors_from_bytes(data: bytes) -> dict:
    """Load SafeTensors from raw bytes (useful for WASM / in-memory models)."""
    if not _LoomLoadSafetensorsFromBytes:
        raise RuntimeError("LoomLoadSafetensorsFromBytes not available in library")
    result = _parse_json(
        _LoomLoadSafetensorsFromBytes(data, len(data))
    )
    return _check(result, "load_safetensors_from_bytes")


def load_safetensors_with_shapes(path: str) -> dict:
    """Load SafeTensors and include shape metadata for each tensor."""
    if not _LoomLoadSafetensorsWithShapes:
        raise RuntimeError("LoomLoadSafetensorsWithShapes not available in library")
    result = _parse_json(_LoomLoadSafetensorsWithShapes(path.encode("utf-8")))
    return _check(result, f"load_safetensors_with_shapes({path!r})")


def load_weights_with_prefixes(network_handle: int, path: str) -> None:
    """
    Load SafeTensors weights into a network using the prefix-tree matcher.

    This is the recommended way to load pre-trained model weights — the
    prefix matcher automatically maps tensor names to layer positions.

    Args:
        network_handle: Target network handle.
        path: Path to .safetensors file.
    """
    if not _LoomLoadWithPrefixes:
        raise RuntimeError("LoomLoadWithPrefixes not available in library")
    result = _parse_json(
        _LoomLoadWithPrefixes(int(network_handle), path.encode("utf-8"))
    )
    _check(result, f"load_weights_with_prefixes({path!r})")


# ---------------------------------------------------------------------------
# DNA / Network Introspection
# ---------------------------------------------------------------------------

def extract_dna(network_handle: int) -> dict:
    """
    Extract a compact DNA fingerprint from the network.

    The DNA captures architecture, weight statistics, and layer signatures —
    useful for comparing networks and detecting divergence.

    Returns:
        NetworkDNA dict.
    """
    if not _LoomExtractDNA:
        raise RuntimeError("LoomExtractDNA not available in library")
    result = _parse_json(_LoomExtractDNA(int(network_handle)))
    return _check(result, "extract_dna")


def compare_dna(dna1: dict, dna2: dict) -> dict:
    """
    Compare two NetworkDNA objects and return a similarity report.

    Args:
        dna1, dna2: DNA dicts from extract_dna().

    Returns:
        Comparison dict with similarity scores per layer.
    """
    if not _LoomCompareDNA:
        raise RuntimeError("LoomCompareDNA not available in library")
    result = _parse_json(
        _LoomCompareDNA(
            json.dumps(dna1).encode("utf-8"),
            json.dumps(dna2).encode("utf-8"),
        )
    )
    return _check(result, "compare_dna")


def extract_network_blueprint(network_handle: int,
                               model_id: str = "model") -> dict:
    """Extract a serializable blueprint of the network topology."""
    if not _LoomExtractNetworkBlueprint:
        raise RuntimeError("LoomExtractNetworkBlueprint not available in library")
    result = _parse_json(
        _LoomExtractNetworkBlueprint(int(network_handle), model_id.encode("utf-8"))
    )
    return _check(result, "extract_network_blueprint")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(path: str) -> int:
    """
    Load a BPE tokenizer from a JSON file (HuggingFace tokenizer.json format).

    Args:
        path: Path to tokenizer.json.

    Returns:
        Tokenizer handle. Free with free_tokenizer().
    """
    if not _LoomLoadTokenizer:
        raise RuntimeError("LoomLoadTokenizer not available in library")
    handle = _LoomLoadTokenizer(path.encode("utf-8"))
    if handle < 0:
        raise RuntimeError(f"Failed to load tokenizer from {path!r}")
    return int(handle)


def tokenize(tokenizer_handle: int, text: str) -> List[int]:
    """
    Encode text to a list of token IDs.

    Args:
        tokenizer_handle: Handle from load_tokenizer().
        text: Input text.

    Returns:
        List of integer token IDs.
    """
    if not _LoomTokenize:
        raise RuntimeError("LoomTokenize not available in library")
    result = _parse_json(_LoomTokenize(int(tokenizer_handle), text.encode("utf-8")))
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(result["error"])
    return result if isinstance(result, list) else []


def detokenize(tokenizer_handle: int, ids: List[int]) -> str:
    """
    Decode a list of token IDs back to text.

    Args:
        tokenizer_handle: Handle from load_tokenizer().
        ids: Token IDs.

    Returns:
        Decoded text string.
    """
    if not _LoomDetokenize:
        raise RuntimeError("LoomDetokenize not available in library")
    ids_json = json.dumps(ids).encode("utf-8")
    ptr = _LoomDetokenize(int(tokenizer_handle), ids_json)
    if not ptr:
        return ""
    return ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8", errors="replace")


def free_tokenizer(handle: int) -> None:
    """Free tokenizer resources."""
    if _LoomFreeTokenizer and handle >= 0:
        _LoomFreeTokenizer(int(handle))


# ---------------------------------------------------------------------------
# Per-layer dispatch (granular control)
# ---------------------------------------------------------------------------

def layer_forward(layer_name: str, network_handle: int, layer_idx: int,
                  state_handle: int) -> dict:
    """
    Run the forward pass for a single named layer.

    Args:
        layer_name: One of the layer type names: "Dense", "RMSNorm",
                    "MHA", "SwiGLU", "CNN1", "CNN2", "CNN3", etc.
        network_handle: Network handle.
        layer_idx: Index within n.Layers.
        state_handle: SystolicState carrying the input tensor.

    Returns:
        Dict with 'pre' and 'post' activation tensors.
    """
    fn = _layer_forward_fns.get(layer_name)
    if fn is None:
        raise RuntimeError(
            f"Loom{layer_name}Forward not available. "
            f"Valid names: {sorted(_layer_forward_fns.keys())}"
        )
    result = _parse_json(fn(int(network_handle), int(layer_idx), int(state_handle)))
    return _check(result, f"layer_forward({layer_name})")


def layer_backward(layer_name: str, network_handle: int, layer_idx: int,
                   state_handle: int) -> dict:
    """
    Run the backward pass for a single named layer.

    Args same as layer_forward().

    Returns:
        Gradient dict.
    """
    fn = _layer_backward_fns.get(layer_name)
    if fn is None:
        raise RuntimeError(
            f"Loom{layer_name}Backward not available. "
            f"Valid names: {sorted(_layer_backward_fns.keys())}"
        )
    result = _parse_json(fn(int(network_handle), int(layer_idx), int(state_handle)))
    return _check(result, f"layer_backward({layer_name})")


# ---------------------------------------------------------------------------
# High-level OOP wrappers
# ---------------------------------------------------------------------------

class Network:
    """
    High-level wrapper around a VolumetricNetwork handle.

    Manages the network handle lifetime and exposes the full M-POLY-VTD API
    through clean method calls.

    Example::

        net = Network({
            "id": "xor_net",
            "depth": 1, "rows": 1, "cols": 3,
            "layers": [
                {"type": "dense", "input_size": 2,  "output_size": 8, "activation": "silu"},
                {"type": "dense", "input_size": 8,  "output_size": 4, "activation": "silu"},
                {"type": "dense", "input_size": 4,  "output_size": 1, "activation": "sigmoid"},
            ]
        })

        output = net.forward([0.0, 1.0])
        print(output)

        net.free()
    """

    def __init__(self, json_config=None, *, _handle: int = None):
        if _handle is not None:
            self._handle = _handle
        elif json_config is not None:
            self._handle = build_network(json_config)
        else:
            raise ValueError("Provide json_config or _handle")

    # --- Inference ---

    def forward(self, inputs: List[float]) -> List[float]:
        """One-shot forward pass (CPU, auto-dtype from first layer)."""
        return sequential_forward(self._handle, inputs)

    def forward_gpu(self, inputs: List[float]) -> List[float]:
        """GPU-accelerated forward pass. Requires init_wgpu() first."""
        return forward_wgpu(self._handle, inputs)

    def forward_token_ids(self, token_ids: List[int]) -> List[float]:
        """GPU LLM-style forward from token IDs."""
        return forward_token_ids_wgpu(self._handle, token_ids)

    def create_state(self, dtype: int = DType.FLOAT32) -> "SystolicState":
        """Create a typed SystolicState bound to this network."""
        return SystolicState(self, dtype=dtype)

    def create_target_prop(self, dtype: int = DType.FLOAT32) -> "TargetPropState":
        """Create a TargetPropState bound to this network."""
        return TargetPropState(self, dtype=dtype)

    # --- Weights / dtype ---

    def morph(self, layer_idx: int, target_dtype: int) -> dict:
        """Morph a layer to a new numerical type."""
        return morph_layer(self._handle, layer_idx, target_dtype)

    def morph_all(self, target_dtype: int) -> None:
        """Morph all layers to the same numerical type."""
        info = self.info()
        for i in range(info.get("total_layers", 0)):
            morph_layer(self._handle, i, target_dtype)

    # --- Model I/O ---

    def load_safetensors(self, path: str) -> None:
        """Load pre-trained weights from a SafeTensors file."""
        load_weights_with_prefixes(self._handle, path)

    # --- GPU ---

    def init_gpu(self) -> dict:
        """Initialize WebGPU context."""
        return init_wgpu(self._handle)

    def to_gpu(self) -> dict:
        """Upload weights to GPU VRAM."""
        return sync_to_gpu(self._handle)

    def to_cpu(self) -> dict:
        """Download weights from GPU to CPU."""
        return sync_to_cpu(self._handle)

    # --- Introspection ---

    def info(self) -> dict:
        """Return network metadata."""
        return get_network_info(self._handle)

    def layer_info(self, layer_idx: int) -> dict:
        """Return telemetry for a specific layer."""
        return get_layer_telemetry(self._handle, layer_idx)

    def dna(self) -> dict:
        """Extract network DNA fingerprint."""
        return extract_dna(self._handle)

    def blueprint(self, model_id: str = "model") -> dict:
        """Extract serializable architecture blueprint."""
        return extract_network_blueprint(self._handle, model_id)

    def methods(self) -> List[str]:
        """List exported C-ABI methods."""
        return get_available_methods()

    # --- Resource management ---

    def free(self) -> None:
        """Release native resources."""
        if self._handle >= 0:
            free_network(self._handle)
            self._handle = -1

    def __del__(self):
        if hasattr(self, "_handle") and self._handle >= 0:
            free_network(self._handle)
            self._handle = -1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.free()

    @classmethod
    def from_file(cls, path: str) -> "Network":
        """Load a network from a SafeTensors or universal model file."""
        return cls(_handle=load_network(path))

    @property
    def handle(self) -> int:
        return self._handle

    def __repr__(self) -> str:
        try:
            info = self.info()
            return (f"Network(layers={info.get('total_layers', '?')}, "
                    f"grid={info.get('grid', '?')})")
        except Exception:
            return f"Network(handle={self._handle})"


class SystolicState:
    """
    Typed execution context for a single forward/backward pass.

    Holds per-layer activation tensors and supports the full training loop.
    Each training step should use a fresh state (or reuse one with set_input).

    Example::

        with net.create_state(DType.FLOAT32) as state:
            state.set_input([0.0, 1.0])
            state.step(capture_history=True)
            output = state.output(layer_idx=2)

            grad = state.compute_loss_gradient([1.0], loss_type="mse")
            grad_in = state.backward(grad)
            state.apply_gradients(net, layer_idx=2, grad_weights=grad, lr=0.01)
    """

    def __init__(self, network: Network, dtype: int = DType.FLOAT32):
        self._net = network
        self._handle = create_systolic_state(network.handle, dtype)
        self.dtype = dtype

    def set_input(self, data: List[float]) -> None:
        """Load input data before stepping."""
        set_input(self._handle, data)

    def step(self, capture_history: bool = False) -> int:
        """Run forward pass. Returns duration in nanoseconds."""
        return systolic_step(self._net.handle, self._handle, capture_history)

    def output(self, layer_idx: int) -> List[float]:
        """Get output from a specific layer after step()."""
        return get_output(self._handle, layer_idx)

    def backward(self, grad_output: List[float]) -> List[float]:
        """Run backpropagation. Requires step(capture_history=True)."""
        return systolic_backward(self._net.handle, self._handle, grad_output)

    def compute_loss_gradient(self, targets: List[float],
                               loss_type: str = "mse") -> List[float]:
        """Compute loss gradient between output and targets."""
        return compute_loss_gradient(self._handle, targets, loss_type)

    def apply_gradients(self, network: Network, layer_idx: int,
                        grad_weights: List[float], lr: float) -> None:
        """Apply weight gradients to a layer."""
        apply_gradients(network.handle, layer_idx, grad_weights, lr)

    def apply_target_prop(self, target: List[float],
                          learning_rate: float) -> None:
        """Apply target propagation update."""
        apply_target_prop(self._net.handle, self._handle, target, learning_rate)

    def free(self) -> None:
        """Release native resources."""
        if self._handle >= 0:
            free_systolic_state(self._handle)
            self._handle = -1

    def __del__(self):
        if hasattr(self, "_handle") and self._handle >= 0:
            free_systolic_state(self._handle)
            self._handle = -1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.free()


class TargetPropState:
    """
    Explicit Target Propagation session.

    Uses gap-based local learning rules instead of backpropagation —
    suitable for online learning, non-differentiable layers, and
    architectures where gradient flow is impractical.

    Example::

        with net.create_target_prop(DType.FLOAT32) as tp:
            output = tp.forward([0.0, 1.0])
            tp.backward([1.0])            # or tp.backward_chain_rule([1.0], lr=0.01)
    """

    def __init__(self, network: Network, dtype: int = DType.FLOAT32):
        self._net = network
        self._handle = create_target_prop_state(network.handle, dtype)
        self.dtype = dtype

    def forward(self, inputs: List[float]) -> List[float]:
        """Run forward pass and store activations."""
        return target_prop_forward(self._net.handle, self._handle, inputs)

    def backward(self, target: List[float]) -> List[float]:
        """Apply standard target propagation update."""
        return target_prop_backward(self._net.handle, self._handle, target)

    def backward_chain_rule(self, target: List[float],
                             learning_rate: float) -> List[float]:
        """Apply hybrid chain-rule + target propagation update."""
        return target_prop_backward_chain_rule(
            self._net.handle, self._handle, target, learning_rate
        )

    def free(self) -> None:
        if self._handle >= 0:
            self._handle = -1

    def __del__(self):
        if hasattr(self, "_handle") and self._handle >= 0:
            self._handle = -1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.free()


class Tokenizer:
    """
    BPE tokenizer backed by Loom's native tokenizer engine.

    Loads HuggingFace-compatible tokenizer.json files and provides
    fast encode/decode for LLM inference.

    Example::

        tok = Tokenizer("path/to/tokenizer.json")
        ids = tok.encode("Hello, world!")
        text = tok.decode(ids)
        tok.free()
    """

    def __init__(self, path: str):
        self._handle = load_tokenizer(path)

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return tokenize(self._handle, text)

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        return detokenize(self._handle, ids)

    def free(self) -> None:
        if self._handle >= 0:
            free_tokenizer(self._handle)
            self._handle = -1

    def __del__(self):
        if hasattr(self, "_handle") and self._handle >= 0:
            free_tokenizer(self._handle)
            self._handle = -1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.free()


# ---------------------------------------------------------------------------
# Convenience training loop helper
# ---------------------------------------------------------------------------

def train_network(
    network: Network,
    inputs: List[List[float]],
    targets: List[List[float]],
    *,
    epochs: int = 10,
    learning_rate: float = 0.01,
    loss_type: str = "mse",
    dtype: int = DType.FLOAT32,
    verbose: bool = False,
) -> List[float]:
    """
    Train a network for a fixed number of epochs using systolic execution.

    Args:
        network: Network instance.
        inputs: List of input vectors.
        targets: List of target vectors (same length as inputs).
        epochs: Number of full passes over the dataset.
        learning_rate: SGD step size.
        loss_type: Loss function ("mse", "cross_entropy", "mae").
        dtype: Numerical type to use during training.
        verbose: Print loss every epoch.

    Returns:
        List of per-epoch mean losses.

    Example::

        losses = train_network(net, X_train, y_train, epochs=50, lr=0.01)
    """
    if len(inputs) != len(targets):
        raise ValueError("inputs and targets must have the same length")

    info = network.info()
    n_layers = info.get("total_layers", 1)
    last_layer = n_layers - 1

    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        with network.create_state(dtype) as state:
            for inp, tgt in zip(inputs, targets):
                # Forward
                state.set_input(inp)
                state.step(capture_history=True)
                output = state.output(last_layer)

                # MSE loss for logging
                loss = sum((o - t) ** 2 for o, t in zip(output, tgt)) / max(len(tgt), 1)
                total_loss += loss

                # Compute gradient and backprop
                grad = state.compute_loss_gradient(tgt, loss_type=loss_type)
                grad_in = state.backward(grad)

                # Update all layers
                for li in range(n_layers):
                    apply_gradients(network.handle, li, grad, learning_rate)

        mean_loss = total_loss / max(len(inputs), 1)
        epoch_losses.append(mean_loss)
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}  loss={mean_loss:.6f}")

    return epoch_losses
