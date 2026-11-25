from pmos import pnes


def main():
    pnes.ppu.ppu_off()

    pnes.ppu.set_background_palette(
        15,
        # greys
        [0, 16, 48],
    )

    # This can also be a global python var.
    # It makes no difference for the NES program since it is a compile-time constant
    text = "Hello World!"

    # This is a compile-time loop and therefore will be compiled as unrolled.
    # If you want it to compile as an actual loop, then `text` needs to be
    # an actual SSA `Value` and replace this loop with something like `pnes.InfinitLoop`.
    # But the functionality for that kind of loop does not exist in `pnes` yet.
    for i, t in enumerate(text):
        # `ord` gets the ascii value of a character,
        # which, by construction, is the index of the corresponding
        # tile index in `Alpha.chr`
        pnes.ppu.set_tile_at(ord(t), 10 + i, 14)

    pnes.ppu.ppu_on()

    with pnes.InfiniteLoopOp() as loop:
        pass

    return pnes.create_index(0) 
