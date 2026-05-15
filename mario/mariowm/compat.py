from __future__ import annotations


def patch_legacy_mario_stack() -> None:
    """Patch older gym-super-mario-bros/nes-py assumptions for NumPy 2.x.

    Our code exposes a Gymnasium-style API, but gym-super-mario-bros internally
    still imports old Gym and uses uint8 arithmetic that overflows on NumPy 2.
    """

    import numpy as np

    if not hasattr(np, "bool8"):
        np.bool8 = np.bool_

    import nes_py._rom as nes_rom

    nes_rom.ROM.prg_rom_size = property(lambda self: int(16 * int(self.header[4])))
    nes_rom.ROM.chr_rom_size = property(lambda self: int(8 * int(self.header[5])))

    import gym_super_mario_bros.smb_env as smb_env

    cls = smb_env.SuperMarioBrosEnv
    cls._area = property(lambda self: int(self.ram[0x0760]) + 1)
    cls._level = property(lambda self: int(self.ram[0x075F]) * 4 + int(self.ram[0x075C]))
    cls._life = property(lambda self: int(self.ram[0x075A]))
    cls._player_state = property(lambda self: int(self.ram[0x000E]))
    cls._stage = property(lambda self: int(self.ram[0x075C]) + 1)
    cls._world = property(lambda self: int(self.ram[0x075F]) + 1)
    cls._x_position = property(lambda self: int(self.ram[0x006D]) * 0x100 + int(self.ram[0x0086]))
    cls._y_pixel = property(lambda self: int(self.ram[0x03B8]))
    cls._y_viewport = property(lambda self: int(self.ram[0x00B5]))

    def _left_x_position(self):
        return (int(self.ram[0x0086]) - int(self.ram[0x071C])) % 256

    def _y_position(self):
        if self._y_viewport < 1:
            return 255 + (255 - self._y_pixel)
        return 255 - self._y_pixel

    cls._left_x_position = property(_left_x_position)
    cls._y_position = property(_y_position)
