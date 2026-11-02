import argparse
import importlib.util
from pathlib import Path
import sys

from mlir import ir as _ir
from mlir.dialects import func


"""
This script compiles a pmos program to MLIR.

Right now, it is hardcoded to compile to an MLIR module
for the NES target.  The intention is to extend this to
any MOS platform.

This tool expects a specific format for the input code.

The first section of code should be all the globals and the
second section contains the main function defining the program.

"""

def main(file_path: Path, output_path: Path):
    with _ir.Context() as ctx, _ir.Location.unknown():
        module = _ir.Module.parse("""
            module attributes {
            // Specifying the 6502 MOS target triple and typical 8-bit data layout
            llvm.target_triple = "mos-nes-nrom",
            llvm.data_layout = "e-m:e-p:16:8-p1:8:8-i16:8-i32:8-i64:8-f32:8-f64:8-a:8-Fi8-n8",
            dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 16>>
            } {}
            """)

        with _ir.InsertionPoint(module.body):
            spec = importlib.util.spec_from_file_location("module.name", file_path)
            program_module = importlib.util.module_from_spec(spec)
            sys.modules["module.name"] = program_module
            spec.loader.exec_module(program_module)

            func.FuncOp.from_py_func()(program_module.main)

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(str(module))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="The path to the file to compile")
    parser.add_argument("output_path", help="The path of the destination file")
    args = parser.parse_args()

    main(Path(args.file_path), Path(args.output_path))
