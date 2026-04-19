#!/bin/bash
# 
# Installation script to set up the Python environment for the fine-tuning workshop
# using uv (https://astral.sh/uv).
#
# Supports: macOS (MPS/CPU), Linux x86_64 CUDA, Linux aarch64 CUDA (DGX Spark / Grace Hopper)

set -e # Exit on error

VENV_NAME=".venv"
PYTHON_VERSION="3.12.13"

# ==============================================================================
# Helper Functions for macOS vllm-metal Installation
# ==============================================================================
fetch_latest_release() {
    local repo_owner="$1"
    local repo_name="$2"

    echo ">>> Fetching latest release for ${repo_owner}/${repo_name}..." >&2
    local latest_release_url="https://api.github.com/repos/${repo_owner}/${repo_name}/releases/latest"
    local release_data

    if ! release_data=$(curl -fsSL "$latest_release_url" 2>&1); then
        echo "Error: Failed to fetch release information." >&2
        echo "Please check your internet connection and try again." >&2
        exit 1
    fi

    if [[ -z "$release_data" ]] || [[ "$release_data" == *"Not Found"* ]]; then
        echo "Error: No releases found for this repository." >&2
        exit 1
    fi

    echo "$release_data"
}

extract_wheel_url() {
    local release_data="$1"
    local python_exec="$2"

    "$python_exec" -c "
import sys
import json
try:
    data = json.loads('''$release_data''')
    assets = data.get('assets',[])
    for asset in assets:
        name = asset.get('name', '')
        if name.endswith('.whl'):
            print(asset.get('browser_download_url', ''))
            break
except Exception as e:
    print('', file=sys.stderr)
"
}

download_and_install_wheel() {
    local wheel_url="$1"
    local package_name="$2"
    local python_exec="$3"

    local wheel_name
    wheel_name=$(basename "$wheel_url")
    echo ">>> Latest release found: $wheel_name"

    local tmp_dir
    tmp_dir=$(mktemp -d)

    echo ">>> Downloading wheel..."
    local wheel_path="$tmp_dir/$wheel_name"

    if ! curl -fsSL "$wheel_url" -o "$wheel_path"; then
        echo "Error: Failed to download wheel." >&2
        rm -rf "$tmp_dir"
        exit 1
    fi

    echo ">>> Installing ${package_name}..."
    if ! uv pip install --python "$python_exec" "$wheel_path"; then
        echo "Error: Failed to install ${package_name}." >&2
        rm -rf "$tmp_dir"
        exit 1
    fi

    rm -rf "$tmp_dir"
    echo ">>> Successfully installed ${package_name}"
}
# ==============================================================================

# 1. Check if uv is installed
if ! command -v uv &>/dev/null; then
    echo "Error: uv is not installed. Please install it first (e.g., 'curl -LsSf https://astral.sh/uv/install.sh | sh')"
    exit 1
fi

# Detect OS and GPU
OS_TYPE=$(uname -s)
ARCH=$(uname -m)
HAS_NVIDIA_GPU=false
if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
    HAS_NVIDIA_GPU=true
fi

echo ">>> Detected OS: $OS_TYPE"
echo ">>> Detected Architecture: $ARCH"
echo ">>> NVIDIA GPU Found: $HAS_NVIDIA_GPU"

echo ">>> Creating virtual environment '$VENV_NAME' with Python $PYTHON_VERSION..."
uv venv "$VENV_NAME" --python "$PYTHON_VERSION" --clear

echo ">>> Activating virtual environment and installing dependencies..."
# Use an absolute path so it doesn't break if we change directories
VENV_PYTHON="$(pwd)/$VENV_NAME/bin/python"

# 2. Install core ML libraries based on hardware
if [ "$OS_TYPE" == "Darwin" ]; then
    echo ">>> Installing macOS-optimized (MPS/CPU) stack..."
    uv pip install -U \
        --python "$VENV_PYTHON" \
        "torch" \
        "torchvision" \
        "torchaudio" \
        "transformers" \
        "trl" \
        "peft" \
        "accelerate" \
        "bitsandbytes" \
        "datasets" \
        "evaluate" \
        "google-tunix>=0.1.6" \
        "jax"
    
    echo "⚠️  Note: 'xformers' and 'unsloth' are primarily for Linux/CUDA and will be skipped on macOS."

    # Detect Apple Silicon to install vllm-metal
    if [[ "$ARCH" == 'arm64' ]]; then
        echo ">>> Apple Silicon detected. Installing vLLM and vllm-metal..."
        
        VLLM_V="0.8.5"
        URL_BASE="https://github.com/vllm-project/vllm/releases/download"
        FILENAME="vllm-$VLLM_V.tar.gz"
        
        echo ">>> Fetching vLLM source ($FILENAME)..."
        curl -OL "$URL_BASE/v$VLLM_V/$FILENAME"
        tar xf "$FILENAME"
        cd "vllm-$VLLM_V"
        
        echo ">>> Installing build dependencies..."
        uv pip install --python "$VENV_PYTHON" ninja cmake setuptools_scm grpcio-tools
        
        uv pip install --python "$VENV_PYTHON" setuptools numba
        
        VENV_BIN_DIR="$(dirname "$VENV_PYTHON")"
        ln -sf /opt/homebrew/bin/ninja "$VENV_BIN_DIR/ninja"
        ln -sf /opt/homebrew/bin/cmake "$VENV_BIN_DIR/cmake"
        
        export PATH="$VENV_BIN_DIR:/opt/homebrew/bin:$PATH"
        export CMAKE_MAKE_PROGRAM="$VENV_BIN_DIR/ninja"
        export CMAKE_GENERATOR="Ninja"
        
        cd ..
        rm -rf "vllm-$VLLM_V"*
        
        REPO_OWNER="vllm-project"
        REPO_NAME="vllm-metal"
        PACKAGE_NAME="vllm-metal"
        
        RELEASE_DATA=$(fetch_latest_release "$REPO_OWNER" "$REPO_NAME")
        WHEEL_URL=$(extract_wheel_url "$RELEASE_DATA" "$VENV_PYTHON")
        
        if [[ -z "$WHEEL_URL" ]]; then
            echo "Error: No wheel file found in the latest release of vllm-metal." >&2
            exit 1
        fi
        
        download_and_install_wheel "$WHEEL_URL" "$PACKAGE_NAME" "$VENV_PYTHON"
    else
        echo "⚠️  Note: vLLM macOS support requires Apple Silicon (arm64). Skipping vllm-metal."
    fi

elif [ "$HAS_NVIDIA_GPU" = true ]; then
    echo ">>> Installing Linux/CUDA-optimized (NVIDIA GPU) stack..."
    
    # Detect maximum supported CUDA version from nvidia-smi
    CUDA_VERSION_STRING=$(nvidia-smi | grep "CUDA Version" | sed -n 's/.*CUDA Version: \([0-9\.]*\).*/\1/p')
    echo ">>> Detected maximum supported CUDA Version: $CUDA_VERSION_STRING"
    
    CUDA_MAJOR=$(echo "$CUDA_VERSION_STRING" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION_STRING" | cut -d. -f2)
    
    echo ">>> Detected Architecture: $ARCH"

    # ── Map CUDA version to PyTorch index tag ──────────────────────────────
    # Note: For aarch64 (e.g. DGX Spark / Grace Hopper), PyTorch wheels are
    # available on the standard index for cu121, cu124, cu128.
    # We do NOT force cu130 here since those wheels may not exist yet.
    if [ "$CUDA_MAJOR" -ge 13 ]; then
        PT_CU_VERSION="cu130"
        JAX_CUDA_TAG="cuda13"
    elif [ "$CUDA_MAJOR" -eq 12 ]; then
        if [ "$CUDA_MINOR" -ge 8 ]; then
            PT_CU_VERSION="cu128"
            JAX_CUDA_TAG="cuda12"
        elif [ "$CUDA_MINOR" -ge 4 ]; then
            PT_CU_VERSION="cu124"
            JAX_CUDA_TAG="cuda12"
        elif [ "$CUDA_MINOR" -ge 1 ]; then
            PT_CU_VERSION="cu121"
            JAX_CUDA_TAG="cuda12"
        else
            PT_CU_VERSION="cu118"
            JAX_CUDA_TAG="cuda11"
        fi
    elif [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then
        PT_CU_VERSION="cu118"
        JAX_CUDA_TAG="cuda11"
    else
        echo "⚠️  Unsupported/Old CUDA version detected ($CUDA_VERSION_STRING). Falling back to cu118."
        PT_CU_VERSION="cu118"
        JAX_CUDA_TAG="cuda11"
    fi

    echo ">>> Selected PyTorch CUDA index : $PT_CU_VERSION"
    echo ">>> Selected JAX CUDA tag       : $JAX_CUDA_TAG"

    # Define vLLM installation target
    VLLM_TARGET="vllm"
    if [ "$PT_CU_VERSION" == "cu130" ] && [ "$ARCH" == "aarch64" ]; then
        echo ">>> Using optimized vLLM cu130 wheel for aarch64..."
        VLLM_TARGET="https://github.com/vllm-project/vllm/releases/download/v0.19.1/vllm-0.19.1+cu130-cp38-abi3-manylinux_2_35_aarch64.whl"
    fi

    # Install core ML libraries + workshop tools in one go using the dynamically selected index.
    # This ensures consistent dependency resolution (e.g., ensuring vLLM gets the CUDA 
    # version of Torch instead of falling back to the CPU version on PyPI).
    uv pip install -U \
        --python "$VENV_PYTHON" \
        --extra-index-url "https://download.pytorch.org/whl/${PT_CU_VERSION}" \
        --index-strategy unsafe-best-match \
        "torch" \
        "torchvision" \
        "torchaudio" \
        "transformers" \
        "trl" \
        "peft" \
        "accelerate" \
        "bitsandbytes" \
        "datasets" \
        "evaluate" \
        "google-tunix>=0.1.6" \
        "$VLLM_TARGET" \
        "autoawq"             # required to load the Qwen-AWQ judge model

    # Install JAX with the correct CUDA flavour.
    echo ">>> Installing JAX with CUDA support (${JAX_CUDA_TAG})..."
    uv pip install -U \
        --python "$VENV_PYTHON" \
        -f "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html" \
        "jax[${JAX_CUDA_TAG}]"

else
    echo ">>> Installing standard Linux (CPU only) stack..."
    uv pip install -U \
        --python "$VENV_PYTHON" \
        "transformers" \
        "torch" \
        "torchvision" \
        "trl" \
        "peft" \
        "accelerate" \
        "bitsandbytes" \
        "datasets" \
        "evaluate" \
        "google-tunix" \
        "jax"
    
    echo "⚠️  Note: 'xformers', 'unsloth', and 'vLLM' require an NVIDIA GPU and will be skipped."
fi

# 3. Install data science and utility libraries (common to all)
echo ">>> Installing data science and utility libraries..."
uv pip install -U \
    --python "$VENV_PYTHON" \
    "pandas" \
    "numpy" \
    "scikit-learn" \
    "matplotlib" \
    "tqdm" \
    "tenacity" \
    "sentencepiece" \
    "sentence-transformers" \
    "gptqmodel" \
    "optimum" \
    "huggingface_hub" \
    "wikipedia-api" \
    "synthetic-data-kit"

# 4. Install Jupyter for notebook support
echo ">>> Installing Jupyter..."
uv pip install -U \
    --python "$VENV_PYTHON" \
    "jupyter" \
    "ipykernel"

# Register the kernel
"$VENV_PYTHON" -m ipykernel install --user --name "fine-tuning-workshop" --display-name "Python (Fine-Tuning Workshop)"

echo ""
echo "✅ Environment setup complete!"
echo "To activate the environment, run:"
echo "source $VENV_NAME/bin/activate"
echo ""
