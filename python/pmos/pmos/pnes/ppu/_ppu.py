from typing import Literal, NamedTuple, Optional, Sequence

from mlir import ir
from mlir.dialects import anes, arith, scf, llvm

from .._helpers import (
    _check_or_get_i8_value,
    _check_or_get_i16_value,
)

__all__ = [
    "SpriteData",
    "bank_spr",
    "oam_clear",
    "wait_nmi",
    "fill_pal_pattern",
    "fill_tiles_from_rle",
    "global_meta_sprite",
    "get_global_meta_sprite",
    "ppu_off",
    "ppu_on",
    "set_background_palette",
    "set_sprite_palette",
    "set_meta_sprite_at",
]


class SpriteData(NamedTuple):
    """Properties of a sprite (8x8 tile) that define a 1 byte attribute

    Fields:
        `tile` (`int`): The index of the file in CHR file.
        `palette_id` (`int`): One of the 4 palette indices (0-3).
        `behind` (`bool`): Whether the sprite is in the foreground
            or background (behind = True means background).
        `flip_hz` (`bool`): Whether to flip sprite horizontally.
        `flip_vr` (`bool`): Whether to flip sprite vertically.

    """

    tile: int
    palette_id: int = 0
    behind: bool = False
    flip_hz: bool = False
    flip_vr: bool = False


def meta_sprite_type(num_rows: int, num_cols: int) -> ir.Type:
    return ir.Type.parse(f"!anes.meta_sprite_handle<{num_rows}, {num_cols}>")


def meta_sprite_attr(sprite_data: Sequence[Sequence[SpriteData]]) -> ir.Attribute:
    """Create `MetaSprite` attribute from a 2D array of `SpriteData`.

    Args:
        `sprite_data` (`Sequence[Sequence[SpriteData]]`): A 2D array of `SpriteData` values.
            Expected to be rectangular; every inner sequence is of same length.

    """
    num_rows = len(sprite_data)
    num_cols = len(sprite_data[0])
    data_array = [
        f"#anes.sprite_data<{{tile={item.tile}, \
        palette_id={item.palette_id}, \
        behind={str(item.behind).lower()}, \
        flip_hz={str(item.flip_hz).lower()}, \
        flip_vr={str(item.flip_vr).lower()}}}>"
        for sublist in sprite_data
        for item in sublist
    ]
    return ir.Attribute.parse(
        f"#anes.meta_sprite<[{', '.join(data_array)}]> : !anes.meta_sprite_handle<{num_rows}, {num_cols}>"
    )


def bank_spr(bank: int | ir.Value) -> None:
    bank = _check_or_get_i8_value(bank)
    anes.ppu_bank_spr(bank=bank)


def oam_clear():
    anes.ppu_oam_clear()


def wait_nmi():
    anes.ppu_wait_nmi()


def fill_pal_pattern(
    palette_pattern: tuple[int, int, int, int],
    length: int | ir.Value,
    x: int | ir.Value,
    y: int | ir.Value,
) -> None:
    """Fill multiple tiles of the screen with specified palettes for a given length.

    Args:
        `palette_pattern` (`tuple[int, int, int, int]`): The 4 palette ids that
            determine the color scheme of each 32x32 tile.
        `length` (`int | Value`): The number of 32x32 tiles to fill.
        `x` (`int | Value`): The starting `x` coordinate (in units of pixels).
        `y` (`int | Value`): The starting `y` coordinate (in units of pixels).

    """
    if len(palette_pattern) != 4:
        raise ValueError("expected `palette_pattern` to have 4 elements")
    length = _check_or_get_i16_value(length)
    x = _check_or_get_i8_value(x)
    y = _check_or_get_i8_value(y)

    anes.fill_pal_pattern_at(ir.DenseI8ArrayAttr.get(palette_pattern), length, x, y)


def fill_tiles_from_rle(
    source: llvm.GlobalOp | ir.Value,
    nametable: Literal["a", "b", "c", "d"],
    x: int | ir.Value,
    y: int | ir.Value,
) -> None:
    """Fill tiles on the screen from a compressed rle format.

    Args:
        `source` (`llvm.GlobalOp | Value`): The rle data pointer.
        `nametable` (`Literal["a", "b", "c", "d"]`): Which nametable to fill.
        `x` (`int | Value`): The starting `x` coordinate (in units of pixels).
        `y` (`int | Value`): The starting `y` coordinate (in units of pixels).

    """
    if isinstance(source, llvm.GlobalOp):
        source = llvm.mlir_addressof(ir.Type.parse("!llvm.ptr"), source.sym_name.value)
    elif isinstance(source, ir.Value):
        raise TypeError("`Value` type not yet supported")
    else:
        raise TypeError("expected `llvm.GlobalOp` or `Value` type")
    if nametable not in ["a", "b", "c", "d"]:
        raise ValueError("expected nametable to be either: a, b, c, or d")

    x = _check_or_get_i8_value(x)
    y = _check_or_get_i8_value(y)

    anes.fill_tiles_from_rle_at(
        source, x, y, ir.Attribute.parse(f"#anes.nametable_attr<{nametable}>")
    )


def global_meta_sprite(
    name: str, sprite_data: Sequence[Sequence[SpriteData]]
) -> anes.GlobalMetaSpriteOp:
    """Create a global meta-sprite.

    Args:
        `name` (`str`): The symbol name of the global.
        `sprite_data` (`Sequence[Sequence[SpriteData]]`): The 2D array of `SpriteData`
            defining the metasprite.

    Returns:
       `GlobalMetaSpriteOp` op.

    """
    num_rows = len(sprite_data)
    num_cols = len(sprite_data[0])
    return anes.global_meta_sprite(
        name, meta_sprite_type(num_rows, num_cols), value=meta_sprite_attr(sprite_data)
    )


def get_global_meta_sprite(global_op: anes.GlobalMetaSpriteOp) -> ir.Value:
    """Gets a global meta-sprite.

    Args:
        `global_op` (`GlobalMetaSpriteOp`): The operation defining the global metasprite.

    Returns:
        A `Value` containing the meta-sprite.

    """
    return anes.get_global_meta_sprite(
        global_op.result_type.value, global_op.sym_name.value
    )


def ppu_off() -> None:
    """Turns PPU off."""
    anes.PPUOffOp()


def ppu_on() -> None:
    """Turns PPU on."""
    anes.PPUOnOp()


def _set_palette(
    palette_type: str,
    universal_bg_color: int,
    palette_0: Optional[Sequence[int]] = None,
    palette_1: Optional[Sequence[int]] = None,
    palette_2: Optional[Sequence[int]] = None,
    palette_3: Optional[Sequence[int]] = None,
):
    if palette_type not in ["background", "sprite"]:
        raise ValueError("expected `palette_type` to be background or sprite")
    if palette_0 is None:
        palette_0 = [universal_bg_color] * 3
    elif len(palette_0) != 3:
        raise ValueError("expected `palette_0` sequence to have 3 integers")
    if palette_1 is None:
        palette_1 = [universal_bg_color] * 3
    elif len(palette_1) != 3:
        raise ValueError("expected `palette_1` sequence to have 3 integers")
    if palette_2 is None:
        palette_2 = [universal_bg_color] * 3
    elif len(palette_2) != 3:
        raise ValueError("expected `palette_2` sequence to have 3 integers")
    if palette_3 is None:
        palette_3 = [universal_bg_color] * 3
    elif len(palette_3) != 3:
        raise ValueError("expected `palette_3` sequence to have 3 integers")
    anes.set_palette(
        universal_bg_color,
        ir.DenseI8ArrayAttr.get(palette_0),
        ir.DenseI8ArrayAttr.get(palette_1),
        ir.DenseI8ArrayAttr.get(palette_2),
        ir.DenseI8ArrayAttr.get(palette_3),
        ir.Attribute.parse(f"#anes.palette_type_attr<{palette_type}>"),
    )


def set_background_palette(
    universal_bg_color: int,
    palette_0: Optional[Sequence[int]] = None,
    palette_1: Optional[Sequence[int]] = None,
    palette_2: Optional[Sequence[int]] = None,
    palette_3: Optional[Sequence[int]] = None,
) -> None:
    """Sets the palettes for the background.

    There are 4 palettes total, each with 3 colors and a shared universal background color.

    If `palette_i` is `None`, then all three of its colors get set to the
    universal background color

    Args:
        `universal_bg_color` (`int`): The universal background color.
        `palette_0` (`Optional[Sequence[int]]`): Palette 0.
        `palette_1` (`Optional[Sequence[int]]`): Palette 1.
        `palette_2` (`Optional[Sequence[int]]`): Palette 2.
        `palette_3` (`Optional[Sequence[int]]`): Palette 3.
    """
    _set_palette(
        "background", universal_bg_color, palette_0, palette_1, palette_2, palette_3
    )


def set_sprite_palette(
    universal_bg_color: int,
    palette_0: Optional[Sequence[int]] = None,
    palette_1: Optional[Sequence[int]] = None,
    palette_2: Optional[Sequence[int]] = None,
    palette_3: Optional[Sequence[int]] = None,
) -> None:
    """Sets the palettes for the sprites.

    Works the same as set_background_palette but for sprites.

    Note that the universal background color should be shared
    between sprites and background. Setting it for one will
    override it for the other.

    Args:
        `universal_bg_color` (`int`): The universal background color.
        `palette_0` (`Optional[Sequence[int]]`): Palette 0.
        `palette_1` (`Optional[Sequence[int]]`): Palette 1.
        `palette_2` (`Optional[Sequence[int]]`): Palette 2.
        `palette_3` (`Optional[Sequence[int]]`): Palette 3.
    """
    _set_palette(
        "sprite", universal_bg_color, palette_0, palette_1, palette_2, palette_3
    )


def set_meta_sprite_at(
    metasprite: ir.Value, x: int | ir.Value, y: int | ir.Value
) -> None:
    """Set a meta-sprite at the specified location.

    Args:
        `metasprite` (`Value`): The meta-sprite to set.
        `x` (`int | Value`): The `x` coordinate (in units of pixels).
        `y` (`int | Value`): The `y` coordinate (in units of pixels).

    """
    x = _check_or_get_i8_value(x)
    y = _check_or_get_i8_value(y)
    anes.set_meta_sprite_at(metasprite, x, y)
