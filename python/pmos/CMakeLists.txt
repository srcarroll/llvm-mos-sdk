# Find the Python interpreter running on the host system
find_package(Python3 REQUIRED COMPONENTS Interpreter)

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/pmos/"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/pmos")

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

set(LLVM_MOS_MLIR_PACKAGES_DIR ${CMAKE_PREFIX_PATH}/python_packages/mlir_core/mlir)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml.in"
    "${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml"
    @ONLY
)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/mlir" 
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    "${LLVM_MOS_MLIR_PACKAGES_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/mlir"
    COMMENT "Symlinking absolute MLIR package into build tree for relative packaging"
)

add_custom_target(pmos_wheel
    COMMAND ${Python3_EXECUTABLE} -m pip wheel 
        --wheel-dir ${CMAKE_CURRENT_BINARY_DIR}/pmos_wheel
        ${CMAKE_CURRENT_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS 
        "${CMAKE_CURRENT_BINARY_DIR}/mlir"
        "${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml"
    COMMENT "Compiling python wheels into target distribution directory"
)
