#!/bin/bash
# Download ONNX Runtime for the current platform
# Usage: ./scripts/download-ort.sh [version]

set -e

ORT_VERSION="${1:-1.23.0}"
DEST_DIR="deps/onnxruntime"

# Detect platform
OS=$(uname -s)
ARCH=$(uname -m)

case "$OS" in
    Linux)
        case "$ARCH" in
            x86_64)
                ARTIFACT="onnxruntime-linux-x64"
                ;;
            aarch64)
                ARTIFACT="onnxruntime-linux-aarch64"
                ;;
            *)
                echo "Unsupported Linux architecture: $ARCH"
                exit 1
                ;;
        esac
        ;;
    Darwin)
        case "$ARCH" in
            x86_64)
                ARTIFACT="onnxruntime-osx-x86_64"
                ;;
            arm64)
                ARTIFACT="onnxruntime-osx-arm64"
                ;;
            *)
                echo "Unsupported macOS architecture: $ARCH"
                exit 1
                ;;
        esac
        ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ARTIFACT}-${ORT_VERSION}.tgz"

echo "Downloading ONNX Runtime ${ORT_VERSION} for ${ARTIFACT}..."
echo "URL: $URL"

mkdir -p "$DEST_DIR"
curl -L -o /tmp/ort.tgz "$URL"
tar -xzf /tmp/ort.tgz --strip-components=1 -C "$DEST_DIR"
rm -f /tmp/ort.tgz

echo "ONNX Runtime installed to $DEST_DIR"
ls -la "$DEST_DIR/lib/"
