"""
Setup script for luganda-tokenizer.
Handles building the C library if needed.

Metadata is in pyproject.toml (PEP 517/518 compliant).
This file only provides the custom BuildExt command.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional
from setuptools import setup
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    """Custom build command that compiles the C library."""

    def run(self):
        # Check if we need to build the C library
        lib_path = self._find_library()

        if not lib_path:
            self._build_c_library()

        # Continue with normal build
        super().run()

    def _find_library(self) -> Optional[Path]:
        """Check if pre-built library exists."""
        parent_dir = Path(__file__).parent.parent
        lib_names = ['libluganda_tok.so', 'libluganda_tok.dylib', 'libluganda_tok.dll']

        for name in lib_names:
            path = parent_dir / name
            if path.exists():
                return path
        return None

    def _build_c_library(self):
        """Build the C shared library."""
        parent_dir = Path(__file__).parent.parent

        # Check for Makefile
        if not (parent_dir / "Makefile").exists():
            raise RuntimeError(
                "Cannot find C source code. "
                "Please build libluganda_tok.so manually with 'make lib'"
            )

        # Run make
        try:
            result = subprocess.run(
                ['make', 'lib'],
                cwd=parent_dir,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
                check=True
            )
            print("Built C library successfully")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to build C library: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                "Make not found. Please install build tools and build manually: "
                "'make lib'"
            )


# Metadata is in pyproject.toml - this just registers the custom command
setup(cmdclass={'build_ext': BuildExt})
