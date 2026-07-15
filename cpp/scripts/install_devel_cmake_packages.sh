#!/usr/bin/env bash
# Install missing CMake package configs into /opt/robot/devel (no Conan).
set -euo pipefail

DEVEL="${1:-/opt/robot/devel}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${REPO_ROOT}/cpp/cmake/devel-packages"

if [[ ! -d "${DEVEL}" ]]; then
  echo "devel prefix not found: ${DEVEL}" >&2
  exit 1
fi

install -d "${DEVEL}/lib/cmake/mujoco"
install -m 644 "${SRC}/mujoco-config.cmake" "${DEVEL}/lib/cmake/mujoco/"
install -m 644 "${SRC}/mujoco-config-version.cmake" "${DEVEL}/lib/cmake/mujoco/"

echo "[install] mujoco cmake -> ${DEVEL}/lib/cmake/mujoco"
echo "[check] lib: ${DEVEL}/lib/libmujoco.so"
echo "[check] include: ${DEVEL}/include/mujoco/mujoco.h"

test -f "${DEVEL}/lib/libmujoco.so" || {
  echo "ERROR: ${DEVEL}/lib/libmujoco.so missing; install MuJoCo libs first." >&2
  exit 1
}

test -f "${DEVEL}/include/mujoco/mujoco.h" || {
  echo "ERROR: ${DEVEL}/include/mujoco/mujoco.h missing; install MuJoCo headers first." >&2
  exit 1
}

echo "Done."
