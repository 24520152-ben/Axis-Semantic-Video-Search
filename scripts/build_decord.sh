#!/bin/sh
# Build NVDEC support without installing a driver in the Modal image.
set -eu

build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT

git clone --quiet https://github.com/dmlc/decord.git "$build_dir/decord"
git -C "$build_dir/decord" checkout --quiet d2e56190286ae394032a8141885f76d5372bd44b
git -C "$build_dir/decord" submodule update --init --recursive --depth 1

# Only link against this library; runtime uses Modal's injected driver libraries.
curl --fail --location --retry 3 --output "$build_dir/decode.deb" \
    https://security.ubuntu.com/ubuntu/pool/restricted/n/nvidia-graphics-drivers-535-server/libnvidia-decode-535-server_535.161.08-0ubuntu2_amd64.deb
printf '%s  %s\n' 77cef31182db79cded803e2ad3d13b07ebbb1890f57bb5284c159734683743a9 \
    "$build_dir/decode.deb" | sha256sum --check -
dpkg-deb --extract "$build_dir/decode.deb" "$build_dir/driver"
nvcuvid=$(find "$build_dir/driver" -name libnvcuvid.so.1 -print -quit)
test -n "$nvcuvid"

cmake -S "$build_dir/decord" -B "$build_dir/decord/build" \
    -DUSE_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    '-DCMAKE_CUDA_ARCHITECTURES=75;86' \
    -DCUDA_NVCUVID_LIBRARY="$nvcuvid" \
    -DCMAKE_SKIP_RPATH=ON
cmake --build "$build_dir/decord/build" --parallel 4
/.uv/uv pip install --python /.uv/.venv/bin/python --force-reinstall --no-deps \
    "$build_dir/decord/python"
