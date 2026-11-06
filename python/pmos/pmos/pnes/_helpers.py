from mlir.dialects import arith
from mlir import ir


def _check_or_get_value_of_int_type(val: int | ir.Value, width: int) -> ir.Value:
    if isinstance(val, int):
        return arith.constant(ir.IntegerType.get_signless(width), val)
    elif isinstance(val, ir.Value):
        if val.type != ir.IntegerType.get_signless(width):
            raise TypeError(f"expected `val` to be i{width} type")
        return val
    else:
        raise TypeError("expected `val` to be `int` or `Value`")


def _check_or_get_i1_value(val: int | ir.Value) -> ir.Value:
    return _check_or_get_value_of_int_type(val, 1)


def _check_or_get_i8_value(val: int | ir.Value) -> ir.Value:
    return _check_or_get_value_of_int_type(val, 8)


def _check_or_get_i16_value(val: int | ir.Value) -> ir.Value:
    return _check_or_get_value_of_int_type(val, 16)
