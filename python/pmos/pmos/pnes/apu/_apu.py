from mlir import ir
from mlir.dialects import anes, arith

from .._helpers import (
    _check_or_get_value_of_int_type,
    _cast_or_get_value_of_int_type,
    _check_or_get_i1_value,
    _check_or_get_i8_value,
    _check_or_get_i16_value,
)

__all__ = ["set_pulse_params", "set_triangle_params", "set_status"]


def set_status(
    *,
    pulse0: bool = False,
    pulse1: bool = False,
    triangle: bool = False,
    noise: bool = False,
    dmc: bool = False,
) -> None:
    anes.apu_set_status(
        pulse0=pulse0, pulse1=pulse1, triangle=triangle, noise=noise, dmc=dmc
    )


def set_pulse_params(
    *,
    pulse_width: int | ir.Value | None = None,
    env_loop: bool | ir.Value | None = None,
    const_volume: bool | ir.Value | None = None,
    volume_env_divider: int | ir.Value | None = None,
    sweep_enable: bool | ir.Value | None = None,
    sweep_period: int | ir.Value | None = None,
    negate_shift: bool | ir.Value | None = None,
    shift: int | ir.Value | None = None,
    period: int | ir.Value | None = None,
    length_counter: int | ir.Value | None = None,
):
    """Set triangle channel parameters.

    See https://www.nesdev.org/wiki/APU_Pulse for more details.

    In some cases, multiple arguments comprise a single register.  When one
    parameter in a register is specified but another parameter in the same
    register is not, then that parameter is always set to 0.  In other words,
    setting individual parameters in a register is destructive.  There is currently no
    way to avoid this automatically since APU registers are write only.  If a user
    wants to keep one parameter while changing another then it is their responsibilty
    to keep track of the value in memory when writing their own program.

    Args:
        `pulse_width` (`int | Value`): Pulse width.
        `env_loop` (`bool | Value`): Length counter halt flag and envelope loop flag.
        `const_volume` (`bool | Value`): Constant volume flag (override envelope output with V).
        `volume_env_divider` (`int | Value`): Channel volume value and envelope divider reload value.

        `sweep_enable` (`bool | Value`): Sweep enable flag.
        `sweep_period` (`int | Value`): Clock rate.
        `negate_shift` (`bool | Value`): Negate flag. `false` to add, `true` to subtract from period.
        `shift` (`int | Value`): Bit shift count.

        `period` (`int | Value`): The period of wave.  This argument is unique in that it spans multiple
            registers. The lower 8 bits comprise an entire register, while the upper 3 bits are contained
            in the same register as `length_counter`. So passing this argument will result
            in 2 registers being written to.

        `length_counter` (`int | Value`): Length counter load value.

    """
    if pulse_width is not None:
        pulse_width = _check_or_get_value_of_int_type(pulse_width, 2)
    if env_loop is not None:
        env_loop = _check_or_get_i1_value(env_loop)
    if const_volume is not None:
        const_volume = _check_or_get_i1_value(const_volume)
    if volume_env_divider is not None:
        volume_env_divider = _check_or_get_i8_value(volume_env_divider)
    if sweep_enable is not None:
        sweep_enable = _check_or_get_i1_value(sweep_enable)
    if sweep_period is not None:
        sweep_period = _check_or_get_i8_value(sweep_period)
    if negate_shift is not None:
        negate_shift = _check_or_get_i1_value(negate_shift)
    if shift is not None:
        shift = _check_or_get_i8_value(shift)
    if period is not None:
        period = _check_or_get_i16_value(period)
    if length_counter is not None:
        length_counter = _check_or_get_i8_value(length_counter)

    return anes.apu_set_pulse_parameters(
        duty=pulse_width,
        env_loop=env_loop,
        const_volume=const_volume,
        env_divider=volume_env_divider,
        sweep_enable=sweep_enable,
        sweep_period=sweep_period,
        negate_shift=negate_shift,
        shift=shift,
        period=period,
        length_counter=length_counter,
    )


def set_triangle_params(
    *,
    counter_control: bool | ir.Value | None = None,
    linear_counter: int | ir.Value | None = None,
    length_counter: int | ir.Value | None = None,
    period: int | ir.Value | None = None,
):
    """Set triangle channel parameters.

    See https://www.nesdev.org/wiki/APU_Triangle for more details.

    In some cases, multiple arguments comprise a single register.  When one
    parameter in a register is specified but another parameter in the same
    register is not, then that parameter is always set to 0.  In other words,
    setting individual parameters in a register is destructive.  There is currently no
    way to avoid this automatically since APU registers are write only.  If a user
    wants to keep one parameter while changing another then it is their responsibilty
    to keep track of the value in memory when writing their own program.

    Args:
        `counter_control` (`bool | Value`): Linear counter control flag and length counter halt flag.
        `linear_counter` (`int | Value`): Linear counter reload value.
        `length_counter` (`int | Value`): Length counter load value.
        `period` (`int | Value`): The period of wave.  This argument is unique in that it spans multiple
            registers. The lower 8 bits comprise an entire register, while the upper 3 bits are contained
            in the same register as `length_counter`. So passing this argument will result
            in 2 registers being written to.

    """
    if counter_control is not None:
        counter_control = _check_or_get_i1_value(counter_control)
    if linear_counter is not None:
        linear_counter = _check_or_get_value_of_int_type(linear_counter, 7)
    if length_counter is not None:
        length_counter = _check_or_get_value_of_int_type(length_counter, 5)

    period_low = None
    period_high = None
    if period is not None:
        period = _cast_or_get_value_of_int_type(period, 16)
        i8_type = ir.IntegerType.get_signless(8)
        if isinstance(period, int):
            period_low = arith.constant(i8_type, period & 255)
            period_high = arith.constant(i8_type, (period >> 8) & 255)
        else:
            period_low = arith.trunci(i8_type, period)
            shift_amount = arith.constant(ir.IntegerType.get_signless(16), 8)
            shifted = arith.shrui(period, shift_amount)
            period_high = arith.trunci(ir.IntegerType.get_signless(3), shifted)

    return anes.apu_set_triangle_parameters(
        counter_control=counter_control,
        linear_counter=linear_counter,
        length_counter=length_counter,
        period_low=period_low,
        period_high=period_high,
    )
