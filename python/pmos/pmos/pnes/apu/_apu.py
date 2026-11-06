from mlir import ir
from mlir.dialects import anes

from .._helpers import (
    _check_or_get_i1_value,
    _check_or_get_i8_value,
    _check_or_get_i16_value,
)

__all__ = ["set_pulse_param", "set_status"]


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


def set_pulse_param(
    *,
    duty: int | ir.Value | None = None,
    env_loop: bool | ir.Value | None = None,
    const_volume: bool | ir.Value | None = None,
    env_divider: int | ir.Value | None = None,
    sweep_enable: bool | ir.Value | None = None,
    sweep_period: int | ir.Value | None = None,
    negate_shift: bool | ir.Value | None = None,
    shift: int | ir.Value | None = None,
    period: int | ir.Value | None = None,
    length_counter: int | ir.Value | None = None,
):
    if duty is not None:
        duty = _check_or_get_i8_value(duty)
    if env_loop is not None:
        env_loop = _check_or_get_i1_value(env_loop)
    if const_volume is not None:
        const_volume = _check_or_get_i1_value(const_volume)
    if env_divider is not None:
        env_divider = _check_or_get_i8_value(env_divider)
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

    return anes.apu_create_pulse_parameters(
        duty=duty,
        env_loop=env_loop,
        const_volume=const_volume,
        env_divider=env_divider,
        sweep_enable=sweep_enable,
        sweep_period=sweep_period,
        negate_shift=negate_shift,
        shift=shift,
        period=period,
        length_counter=length_counter,
    )
