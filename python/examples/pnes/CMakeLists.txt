cmake_minimum_required(VERSION 3.18)

set(LLVM_MOS_PLATFORM nes-nrom)
find_package(llvm-mos-sdk REQUIRED)

project(06_color LANGUAGES C ASM)

set(CMAKE_EXECUTABLE_SUFFIX .nes)

include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/pmos-toolchain.cmake")
compile_mlir(nes_example)
