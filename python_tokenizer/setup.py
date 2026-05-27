"""
Setup script for luganda-tokenizer.
Handles building the C library if needed.
"""

import os
import subprocess
import sys
from pathlib import Path
from setuptools import setup, Extension
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
    
    def _find_library(self) -> Path:
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
                "Please build libluganda_tok.so manually with 'make libluganda_tok.so'"
            )
        
        # Run make
        try:
            result = subprocess.run(
                ['make', 'libluganda_tok.so'],
                cwd=parent_dir,
                capture_output=True,
                text=True,
                check=True
            )
            print("Built C library successfully")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to build C library: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                "Make not found. Please install build tools and build manually: "
                "'make libluganda_tok.so'"
            )


# Read version from __init__.py
here = Path(__file__).parent.resolve()
version_file = here / "__init__.py"
version = "1.0.0"
if version_file.exists():
    with open(version_file) as f:
        for line in f:
            if line.startswith('__version__'):
                version = line.split('=')[1].strip().strip('"').strip("'")
                break

# Long description from README
readme_file = here.parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='luganda-tokenizer',
    version=version,
    description='High-performance Luganda tokenizer with Python bindings',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='AI Assisted Port',
    python_requires='>=3.8',
    packages=['python_tokenizer'],
    cmdclass={'build_ext': BuildExt},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires=[],
    extras_require={
        'dev': ['pytest', 'pytest-cov', 'black', 'mypy', 'ruff'],
        'numpy': ['numpy>=1.20.0'],
    },
)
