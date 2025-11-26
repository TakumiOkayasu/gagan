#!/bin/bash
# setup_models.sh - Download ML models for GAGAN OCR tool
#
# Models:
#   - ESPCN_x4.pb: Super-resolution model (4x upscaling)
#   - frozen_east_text_detection.pb: EAST text detection model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"

# Model URLs
ESPCN_URL="https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb"
EAST_URL="https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb"

download_model() {
    local url="$1"
    local output="$2"
    local name="$3"

    if [[ -f "$output" ]]; then
        echo "[SKIP] ${name} already exists"
        return 0
    fi

    echo "[DOWNLOAD] ${name}..."
    if curl -fSL --progress-bar -o "$output" "$url"; then
        echo "[OK] ${name}"
    else
        echo "[ERROR] Failed to download ${name}" >&2
        rm -f "$output"
        return 1
    fi
}

main() {
    echo "=== GAGAN Model Setup ==="
    echo ""

    # Create models directory
    mkdir -p "$MODELS_DIR"

    # Download models
    download_model "$ESPCN_URL" "${MODELS_DIR}/ESPCN_x4.pb" "ESPCN_x4.pb (Super-resolution)"
    download_model "$EAST_URL" "${MODELS_DIR}/frozen_east_text_detection.pb" "frozen_east_text_detection.pb (EAST)"

    echo ""
    echo "=== Setup Complete ==="
}

main "$@"
