find_package(LLVM REQUIRED CONFIG)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

function(compile_mlir name)
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${name}.mlir"
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../python/pmos/pmos/scripts/compile.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/${name}.py"
            "${CMAKE_CURRENT_BINARY_DIR}/${name}.mlir"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${name}.py"
        COMMENT "Running compile.py on ${CMAKE_CURRENT_SOURCE_DIR}/${name}.py"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${name}-llvm.mlir"
        COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt "${CMAKE_CURRENT_BINARY_DIR}/${name}.mlir"
                --convert-to-llvm=dynamic=true
                --convert-scf-to-cf
                --convert-cf-to-llvm
                --convert-arith-to-llvm
                --canonicalize
                --cse
                -o "${CMAKE_CURRENT_BINARY_DIR}/${name}-llvm.mlir"
        DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${name}.mlir"
        COMMENT "Running mlir-opt on ${CMAKE_CURRENT_SOURCE_DIR}/${name}.mlir"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${name}.ll"
        COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-translate "${CMAKE_CURRENT_BINARY_DIR}/${name}-llvm.mlir"
                --mlir-to-llvmir
                -o "${CMAKE_CURRENT_BINARY_DIR}/${name}.ll"
        DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${name}-llvm.mlir"
        COMMENT "Running mlir-translate on ${CMAKE_CURRENT_SOURCE_DIR}/${name}-llvm.mlir"
        VERBATIM
    )
    add_custom_target(${name}_llvm ALL
        DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${name}.ll"
    )
    add_executable(${name} "${CMAKE_CURRENT_BINARY_DIR}/${name}.ll" chr-rom.s)
    set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/${name}.ll" PROPERTIES LANGUAGE C)
    target_include_directories(${name} PRIVATE .)
    target_link_libraries(${name} neslib nesdoug)
endfunction()
