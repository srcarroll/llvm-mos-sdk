from typing import Sequence

from mlir import ir
from mlir.dialects import arith, scf, llvm

import numpy as np

__all__ = ["InfiniteLoopOp", "global_array"]


I8Type = ir.IntegerType.get_signless(8)


class InfiniteLoopOp:
    """Create an infinite loop in which to insert operations.

    Infinite loop here is a `scf.while` (do-while) loop with an always true condition.
    Users can insert operations in the "do" branch of the loop.  An iteration
    count SSA value is created and can be accessed from the block argument
    of the "do" block.

    A context manager is implemented to set the insertion point to the start of
    the "do" block when entering the context.  That way, any operation that
    is invoked in the scope of the context will be inserted in the body of
    the "do" branch. Upon exiting the context, the iteration variable is incremented
    by 1, terminated with an `scf.yield`, and insertion point exited setting it
    to after the `scf.while` operation.

    """

    def __init__(self):
        init = arith.constant(I8Type, 0)
        true_val = arith.constant(ir.IntegerType.get_signless(1), True)
        self.while_op = scf.WhileOp([I8Type], [init])
        self.while_op.before.blocks.append()
        iteration = self.while_op.before.blocks[0].add_argument(
            I8Type, ir.Location.unknown()
        )
        with ir.InsertionPoint(self.while_op.before.blocks[0]):
            scf.condition(true_val, [iteration])
        self.while_op.after.blocks.append()
        iteration = self.while_op.after.blocks[0].add_argument(
            I8Type, ir.Location.unknown()
        )
        self.while_op.iteration = iteration

    def __enter__(self):
        self.ip = ir.InsertionPoint(self.while_op.after.blocks[0])
        self.ip.__enter__()
        return self.while_op

    def __exit__(self, exc_type, exc_value, traceback):
        one = arith.constant(I8Type, 1)
        next_val = arith.addi(self.while_op.iteration, one)
        scf.yield_([next_val])
        return self.ip.__exit__(exc_type, exc_value, traceback)


def global_array(
    name: str, array: Sequence[int] | str, *, constant: bool | None = None
):
    """Create an global array of integers or characters (both 8-bit)

    Args:
        `name` (`str`): The symbol name of the global var.
        `array` (`Sequence[int] | str`): The init value of the global.
            Either an array of integers or a string.
        `constant` (`bool | None`): Whether the array is constant.

    """
    if len(array) == 0:
        value = None
    else:
        if isinstance(array[0], str):
            value = ir.StringAttr.get(array)
        elif isinstance(array[0], int):
            value = ir.DenseElementsAttr.get(np.array(array, dtype=np.int8))
        else:
            raise ValueError("expected `array` to consist of `str` or `int` elements")
    return llvm.mlir_global(
        ir.Type.parse(f"!llvm.array<{len(array)} x i8>"),
        name,
        linkage=ir.Attribute.parse("#llvm.linkage<internal>"),
        value=value,
        alignment=1,
        constant=True,
    )
