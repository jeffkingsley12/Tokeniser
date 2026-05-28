"""
closed_loop_agent.py
====================
Python FFI bindings for the Gemini Linguistic Engine closed-loop controller.
Binds to libgemini.so via ctypes.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from pathlib import Path
from typing import Union

# ── Locate and load the shared library ───────────────────────────────────────

def _find_gemini_lib() -> str:
    """Search for the shared library next to this file, then parent dir, then system path."""
    ext  = "dylib" if sys.platform == "darwin" else "so"
    name = f"libgemini.{ext}"
    here = Path(__file__).parent / name
    if here.exists():
        return str(here)
    parent = Path(__file__).parent.parent / name
    if parent.exists():
        return str(parent)
    # Fall back to whatever the dynamic linker can find.
    return name


def _load_gemini_lib() -> ctypes.CDLL:
    # First find and load libluganda_tok to satisfy libgemini dependency
    from .luganda_tokenizer import _find_lib as _find_tok_lib
    
    # RTLD_GLOBAL is required on Linux/macOS so libgemini can resolve symbols from libluganda_tok
    mode = ctypes.RTLD_GLOBAL if sys.platform != "win32" else 0
    try:
        ctypes.CDLL(_find_tok_lib(), mode=mode)
    except Exception:
        pass
        
    lib_path = _find_gemini_lib()
    lib = ctypes.CDLL(lib_path)

    # Opaque Context Lifetime
    lib.le_load_mmap.argtypes = [ctypes.c_char_p, ctypes.c_bool]
    lib.le_load_mmap.restype = ctypes.c_void_p

    lib.le_unload_mmap.argtypes = [ctypes.c_void_p]
    lib.le_unload_mmap.restype = None

    # Telemetry Hooks
    lib.engine_get_scc_stability.argtypes = [ctypes.c_void_p]
    lib.engine_get_scc_stability.restype = ctypes.c_float

    lib.engine_get_entropy_delta.argtypes = [ctypes.c_void_p]
    lib.engine_get_entropy_delta.restype = ctypes.c_float

    lib.engine_get_active_region_count.argtypes = [ctypes.c_void_p]
    lib.engine_get_active_region_count.restype = ctypes.c_uint32

    lib.engine_set_ingestion_weight_modifier.argtypes = [ctypes.c_void_p, ctypes.c_float]
    lib.engine_set_ingestion_weight_modifier.restype = None

    # Ingestion Hooks
    lib.le_process_token.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
    lib.le_process_token.restype = ctypes.c_uint32

    # Global Stats
    lib.get_node_count.argtypes = [ctypes.c_void_p]
    lib.get_node_count.restype = ctypes.c_uint32

    lib.get_edge_count.argtypes = [ctypes.c_void_p]
    lib.get_edge_count.restype = ctypes.c_uint32

    lib.get_scc_count.argtypes = [ctypes.c_void_p]
    lib.get_scc_count.restype = ctypes.c_uint32

    lib.get_symbol_count.argtypes = [ctypes.c_void_p]
    lib.get_symbol_count.restype = ctypes.c_uint32

    # Epoch Management
    lib.le_begin_epoch.argtypes = [ctypes.c_void_p]
    lib.le_begin_epoch.restype = None

    # DAWG Promotion
    lib.le_promote_eligible.argtypes = [ctypes.c_void_p]
    lib.le_promote_eligible.restype = ctypes.c_uint32

    # Serialization
    lib.le_save_mmap.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.le_save_mmap.restype = ctypes.c_bool

    return lib


_lib = _load_gemini_lib()

class EngineConfig(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("rho_min", ctypes.c_float),
        ("h_max", ctypes.c_float),
        ("h_forced", ctypes.c_float),
        ("min_freq", ctypes.c_uint32),
        ("promotion_epochs", ctypes.c_uint32),
        ("promotion_budget", ctypes.c_uint32),
    ]

# ── Agent Implementation ──────────────────────────────────────────────────────

class LugandaClosedLoopEngineAgent:
    """
    Ergonomic Python controller agent for the Gemini Cognitive Graph.
    Uses closed-loop feedback telemetry to adaptively throttle data ingestion.
    """

    def __init__(self, model_path: Union[str, Path], writable: bool = True) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model snapshot file not found: {self.model_path}")
            
        path_bytes = os.fsencode(str(self.model_path))
        self._ctx = _lib.le_load_mmap(path_bytes, writable)
        if not self._ctx:
            raise RuntimeError(f"Failed to load Gemini Engine model from {self.model_path}")
            
        self._closed = False
        self._lock = threading.Lock()

    def close(self) -> None:
        """Release the memory-mapped engine context. Safe to call multiple times."""
        with self._lock:
            if not self._closed and self._ctx:
                _lib.le_unload_mmap(self._ctx)
                self._ctx = None
                self._closed = True

    def __enter__(self) -> LugandaClosedLoopEngineAgent:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _check_open(self) -> None:
        if self._closed or not self._ctx:
            raise RuntimeError("Engine context is closed or uninitialized")

    @property
    def rho_min(self) -> float:
        self._check_open()
        cfg = ctypes.cast(self._ctx, ctypes.POINTER(EngineConfig)).contents
        return float(cfg.rho_min)

    @rho_min.setter
    def rho_min(self, value: float) -> None:
        self._check_open()
        cfg = ctypes.cast(self._ctx, ctypes.POINTER(EngineConfig)).contents
        cfg.rho_min = value

    @property
    def h_max(self) -> float:
        self._check_open()
        cfg = ctypes.cast(self._ctx, ctypes.POINTER(EngineConfig)).contents
        return float(cfg.h_max)

    @h_max.setter
    def h_max(self, value: float) -> None:
        self._check_open()
        cfg = ctypes.cast(self._ctx, ctypes.POINTER(EngineConfig)).contents
        cfg.h_max = value

    @property
    def min_freq(self) -> int:
        self._check_open()
        cfg = ctypes.cast(self._ctx, ctypes.POINTER(EngineConfig)).contents
        return int(cfg.min_freq)

    @min_freq.setter
    def min_freq(self, value: int) -> None:
        self._check_open()
        cfg = ctypes.cast(self._ctx, ctypes.POINTER(EngineConfig)).contents
        cfg.min_freq = value

    @property
    def promotion_epochs(self) -> int:
        self._check_open()
        cfg = ctypes.cast(self._ctx, ctypes.POINTER(EngineConfig)).contents
        return int(cfg.promotion_epochs)

    @promotion_epochs.setter
    def promotion_epochs(self, value: int) -> None:
        self._check_open()
        cfg = ctypes.cast(self._ctx, ctypes.POINTER(EngineConfig)).contents
        cfg.promotion_epochs = value

    @property
    def scc_stability(self) -> float:
        """Get the average structural coherence (rho) across active unpromoted SCCs."""
        self._check_open()
        return float(_lib.engine_get_scc_stability(self._ctx))

    @property
    def entropy_delta(self) -> float:
        """Get the real-time difference between current system entropy and previous epoch."""
        self._check_open()
        return float(_lib.engine_get_entropy_delta(self._ctx))

    @property
    def active_region_count(self) -> int:
        """Get the count of active, mutating SCC candidates."""
        self._check_open()
        return int(_lib.engine_get_active_region_count(self._ctx))

    def set_weight_modifier(self, modifier: float) -> None:
        """Set runtime weight multiplier for incoming token / edge additions."""
        self._check_open()
        if modifier < 0.0:
            modifier = 0.0
        _lib.engine_set_ingestion_weight_modifier(self._ctx, ctypes.c_float(modifier))

    def process_token(self, token_id: int, prev_node: int = 0xFFFFFFFF) -> int:
        """Ingest a single token into the engine context and return the new node ID."""
        self._check_open()
        return int(_lib.le_process_token(self._ctx, ctypes.c_uint32(token_id), ctypes.c_uint32(prev_node)))

    def begin_epoch(self) -> None:
        """Manually trigger an epoch transition in the engine."""
        self._check_open()
        _lib.le_begin_epoch(self._ctx)

    def promote_eligible(self) -> int:
        """Evaluate active candidates and promote stable SCCs to DAWG symbols."""
        self._check_open()
        return int(_lib.le_promote_eligible(self._ctx))

    def save(self, path: Union[str, Path, None] = None) -> bool:
        """
        Save the current cognitive graph snapshot to disk.
        If path is not provided, overwrites the model_path.
        """
        self._check_open()
        if path is None:
            path = self.model_path
        else:
            path = Path(path)
        path_bytes = os.fsencode(str(path))
        return bool(_lib.le_save_mmap(self._ctx, path_bytes))

    @property
    def node_count(self) -> int:
        self._check_open()
        return int(_lib.get_node_count(self._ctx))

    @property
    def edge_count(self) -> int:
        self._check_open()
        return int(_lib.get_edge_count(self._ctx))

    @property
    def scc_count(self) -> int:
        self._check_open()
        return int(_lib.get_scc_count(self._ctx))

    @property
    def symbol_count(self) -> int:
        self._check_open()
        return int(_lib.get_symbol_count(self._ctx))

    def compute_crawl_delay(self, base_delay: float) -> float:
        """
        Adaptive closed-loop feedback controller.
        Calculates an adjusted crawling delay (back-off) based on semantic stability and entropy.
        Also dynamically modulates the C-level ingestion weights to protect against noisy bursts.
        """
        stability = self.scc_stability
        ent_delta = self.entropy_delta
        active_regions = self.active_region_count

        # Map low stability (high structural chaos) to a penalty factor
        chaos = (1.0 - stability) if stability > 0.0 else 0.5
        
        # Penalize fast-rising entropy deltas
        ent_factor = max(0.0, ent_delta)
        
        # Calculate exponential back-off multiplier
        multiplier = 1.0 + (chaos * 2.0) + (ent_factor * 5.0) + (active_regions * 0.1)
        multiplier = min(10.0, max(1.0, multiplier))
        
        # Apply the inverse multiplier to ingestion weights (damps noisy inputs)
        weight_modifier = 1.0 / multiplier
        self.set_weight_modifier(weight_modifier)
        
        return base_delay * multiplier

    def __repr__(self) -> str:
        state = "closed" if self._closed else f"nodes={self.node_count}, symbols={self.symbol_count}"
        return f"LugandaClosedLoopEngineAgent({state})"
