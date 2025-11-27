#!/usr/bin/env bash
set -euo pipefail

# Versions and URLs
MSGPACK_VER="7.0.0"
MSGPACK_TARBALL="msgpack-cxx-${MSGPACK_VER}.tar.gz"
MSGPACK_URL="https://github.com/msgpack/msgpack-c/releases/download/cpp-${MSGPACK_VER}/${MSGPACK_TARBALL}"
MSGPACK_DIR="msgpack-cxx-${MSGPACK_VER}"

LIBZMQ_VER="4.3.5"
LIBZMQ_TARBALL="zeromq-${LIBZMQ_VER}.tar.gz"
LIBZMQ_URL="https://github.com/zeromq/libzmq/releases/download/v${LIBZMQ_VER}/zeromq-${LIBZMQ_VER}.tar.gz"
LIBZMQ_DIR="zeromq-${LIBZMQ_VER}"

CPPZMQ_VER="4.11.0"
CPPZMQ_TARBALL="v${CPPZMQ_VER}.tar.gz"
CPPZMQ_URL="https://github.com/zeromq/cppzmq/archive/refs/tags/${CPPZMQ_TARBALL}"
CPPZMQ_DIR="cppzmq-${CPPZMQ_VER}"

download_if_missing() {
  local url="$1"
  local file="$2"
  if [[ -f "$file" ]]; then
    echo "[skip] ${file} already exists, skip download."
  else
    echo "[wget] ${url}"
    wget --no-check-certificate -O "$file" "$url"
  fi
}

extract_tarball() {
  local file="$1"
  echo "[tar ] extracting ${file}"
  tar -zxf "$file"
}

build_install_msgpack() {
  echo "[build] msgpack-c ${MSGPACK_VER}"
  cd "${MSGPACK_DIR}"
  mkdir -p build && cd build
  cmake .. \
    -DMSGPACK_BUILD_EXAMPLES=OFF \
    -DMSGPACK_BUILD_TESTS=OFF \
    -DMSGPACK_USE_BOOST=OFF \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DMSGPACK_ENABLE_SHARED=ON
  make -j"$(nproc)"
  make install
  cd ../..
  echo "[clean] ${MSGPACK_DIR}"
  rm -rf "${MSGPACK_DIR}"
}

build_install_libzmq() {
  echo "[build] libzmq ${LIBZMQ_VER}"
  cd "${LIBZMQ_DIR}"
  mkdir -p build && cd build
  cmake .. \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DCPPZMQ_BUILD_TESTS=OFF
  make -j"$(nproc)" || true
  make install
  cd ../..
  echo "[clean] ${LIBZMQ_DIR}"
  rm -rf "${LIBZMQ_DIR}"
}

build_install_cppzmq() {
  echo "[build] cppzmq ${CPPZMQ_VER}"
  cd "${CPPZMQ_DIR}"
  mkdir -p build && cd build
  cmake .. \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DCPPZMQ_BUILD_TESTS=OFF
  make -j"$(nproc)" || true
  make install
  cd ../..
  echo "[clean] ${CPPZMQ_DIR}"
  rm -rf "${CPPZMQ_DIR}"
}

main() {
  # msgpack-c
  download_if_missing "${MSGPACK_URL}" "${MSGPACK_TARBALL}"
  extract_tarball "${MSGPACK_TARBALL}"
  build_install_msgpack

  # libzmq
  download_if_missing "${LIBZMQ_URL}" "${LIBZMQ_TARBALL}"
  extract_tarball "${LIBZMQ_TARBALL}"
  build_install_libzmq

  # cppzmq
  download_if_missing "${CPPZMQ_URL}" "${CPPZMQ_TARBALL}"
  extract_tarball "${CPPZMQ_TARBALL}"
  build_install_cppzmq
}

main "$@"
