#!/usr/bin/env bash
# Build MicroXRCEAgent from source and install it to /usr/local.
#
# Lives in its own script (not inlined in the Dockerfile) so we can emit
# detailed diagnostics and handle the v3.0.0 superbuild's missing install
# step deterministically, without tangling the whole pipeline into one
# long backslash-continued shell command.

set -euo pipefail

UXRCE_REF="${1:-v3.0.0}"
SRC=/tmp/uxrce
BUILD=/tmp/uxrce/build

echo ">>> Cloning Micro-XRCE-DDS-Agent@${UXRCE_REF}"
git clone --depth 1 --branch "${UXRCE_REF}" \
    https://github.com/eProsima/Micro-XRCE-DDS-Agent.git "${SRC}"

echo ">>> Configuring superbuild"
cmake -S "${SRC}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUAGENT_BUILD_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX=/usr/local

echo ">>> Building (this fetches fast-dds/fast-cdr and compiles the agent)"
cmake --build "${BUILD}" -j"$(nproc)"

echo ">>> Running cmake --install on every inner build tree with an install script"
# The outer superbuild Makefile has no install target, and for v3.0.0 the
# uagent ExternalProject has no install step either. Walk every directory
# with a cmake_install.cmake and install what we can.
INSTALLED_ANYTHING=0
while IFS= read -r -d '' install_script; do
    dir=$(dirname "${install_script}")
    echo "  -> cmake --install ${dir}"
    if cmake --install "${dir}" 2>&1; then
        INSTALLED_ANYTHING=1
    fi
done < <(find "${BUILD}" -name cmake_install.cmake -print0)

echo ">>> Locating MicroXRCEAgent binary in the build tree"
AGENT_BIN=$(find "${BUILD}" -name MicroXRCEAgent -type f -executable | head -n1 || true)
if [[ -z "${AGENT_BIN}" ]]; then
    echo "FATAL: no MicroXRCEAgent binary found in ${BUILD}" >&2
    exit 1
fi
echo "  found: ${AGENT_BIN}"

# Fallback install: copy the binary and libmicroxrcedds_agent.so manually.
# `install -D` handles dest dir creation. We also strip the RPATH so the
# binary resolves its shared libs via ldconfig/LD_LIBRARY_PATH instead of
# the now-deleted /tmp/uxrce paths that the inner cmake embedded.
if [[ ! -x /usr/local/bin/MicroXRCEAgent ]]; then
    echo ">>> Copying MicroXRCEAgent to /usr/local/bin/ (fallback)"
    install -Dm755 "${AGENT_BIN}" /usr/local/bin/MicroXRCEAgent
fi

echo ">>> Copying libmicroxrcedds_agent.so* to /usr/local/lib/"
while IFS= read -r -d '' lib; do
    echo "  -> ${lib}"
    install -Dm755 "${lib}" "/usr/local/lib/$(basename "${lib}")"
done < <(find "${BUILD}" -name 'libmicroxrcedds_agent*.so*' -print0)

echo ">>> Stripping RPATH so the runtime linker uses /usr/local/lib"
patchelf --remove-rpath /usr/local/bin/MicroXRCEAgent || true
for lib in /usr/local/lib/libmicroxrcedds_agent*.so*; do
    [[ -f "${lib}" ]] || continue
    patchelf --remove-rpath "${lib}" || true
done

ldconfig

echo ">>> Verifying install"
which MicroXRCEAgent
MicroXRCEAgent --help 2>&1 | head -5 || true

echo ">>> Cleaning up source tree"
rm -rf "${SRC}"

echo ">>> MicroXRCEAgent install complete"
