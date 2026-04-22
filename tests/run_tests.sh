#!/usr/bin/env bash
# Usage: ./tests/run_tests.sh [--update-baselines]
set -e
cd "$(dirname "$0")/.."

BASE_IMAGE=webgpu-base
IMAGE=ngsolve-webgpu-tests
EXTRA=()

for arg in "$@"; do
    case "$arg" in
        --update-baselines) EXTRA+=(-e UPDATE_BASELINES=1) ;;
        *) echo "Usage: $0 [--update-baselines]"; exit 1 ;;
    esac
done

# Build the webgpu base image from the sibling repo
WEBGPU_DIR="$(cd ../webgpu && pwd)"
echo "==> Building base image (webgpu-base)..."
docker build -f "$WEBGPU_DIR/tests/Dockerfile.base" -t "$BASE_IMAGE" "$WEBGPU_DIR"

echo "==> Building test image (ngsolve-webgpu-tests)..."
docker build -f tests/Dockerfile --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$IMAGE" .

echo "==> Running tests..."
docker run --rm \
    -v "$(pwd)/tests/output:/app/tests/output" \
    -v "$(pwd)/tests/baselines:/app/tests/baselines" \
    "${EXTRA[@]}" \
    "$IMAGE"

echo "==> Done. Screenshots in tests/output/"
if [[ " ${EXTRA[*]} " == *"UPDATE_BASELINES=1"* ]]; then
    echo "==> Baselines updated in tests/baselines/"
fi