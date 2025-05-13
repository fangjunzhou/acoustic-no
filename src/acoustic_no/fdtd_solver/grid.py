import logging
from typing import List, Tuple
import taichi as ti

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@ti.data_oriented
class Grid2D:
    # Staggered grid for pressure and velocity.
    alpha_grid: ti.Field
    p_grid: ti.Field
    v_grid: ti.Field
    vx_grid: ti.Field
    vy_grid: ti.Field
    # Simulation grid size.
    size: Tuple[int, int]
    # Simulation cell size.
    dx: ti.f32
    # Sound wave speed.
    c: ti.f32
    # Time step size.
    dt: ti.f32

    def __init__(self, size: Tuple[int, int], dx: ti.f32, c: ti.f32 = 340) -> None:
        self.size = size
        self.dx = dx
        self.c = c
        self.dt = dx / (ti.sqrt(2) * c)

        sx, sy = size
        self.alpha_grid = ti.field(ti.f32, shape=(sx, sy))
        self.p_grid = ti.field(ti.f32, shape=(sx, sy))
        self.v_grid = ti.field(ti.math.vec2, shape=(sx, sy))
        self.vx_grid = ti.field(ti.f32, shape=(sx + 1, sy))
        self.vy_grid = ti.field(ti.f32, shape=(sx, sy + 1))


def main():
    # Initialize taichi.
    ti.init(arch=ti.gpu)
    # Construct scene.
    grid = Grid2D((512, 512), 0.01, 340)
    logger.info(f"Scene grid shape: {grid.size}.")


if __name__ == "__main__":
    main()
