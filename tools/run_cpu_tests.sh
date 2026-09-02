#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../build
c++ -std=c++17 -O1 -g -Wno-unknown-pragmas -pthread \
    -o ../build/cpu_emu_test cpu_emu_test.cpp cuda_cpu_shim.cpp
../build/cpu_emu_test
