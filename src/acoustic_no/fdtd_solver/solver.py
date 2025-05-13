import argparse
import logging
import pathlib
from typing import List
import taichi as ti
import numpy as np

from acoustic_no.fdtd_solver.grid import Grid2D
from acoustic_no.fdtd_solver.object import Object, Circle


logger = logging.getLogger(__name__)


class Solver2D:
    # Scene grid.
    grid: Grid2D
    # Scene objects.
    objects: List[Object]

    # Solver states.
    damping_grid: ti.Field
    blend_dist: float

    # Simulation states.
    t: ti.f32 = 0

    def __init__(
        self,
        grid: Grid2D,
        damp_factor: float = 0,
        pml: int = 8,
        pml_damp: float = 0.02,
        blend_dist: float = 1,
    ) -> None:
        self.grid = grid
        self.objects = []
        self.damping_grid = ti.field(ti.f32, shape=grid.size)
        self.blend_dist = blend_dist

        @ti.kernel
        def setup_damping_grid():
            sx, sy = self.grid.size
            for i, j in self.damping_grid:
                dl = i
                dr = sx - i
                db = j
                dt = sy - j
                d = ti.math.min(dl, dr, db, dt)
                # PML
                if d < pml:
                    alpha = d / pml
                    self.damping_grid[i, j] = (
                        alpha * damp_factor + (1 - alpha) * pml_damp
                    )
                else:
                    self.damping_grid[i, j] = damp_factor

        setup_damping_grid()

    def rasterize(self, t: float):
        @ti.kernel
        def clear_alpha():
            for i, j in self.grid.alpha_grid:
                self.grid.alpha_grid[i, j] = 0

        clear_alpha()
        for obj in self.objects:
            obj.rasterize_alpha(self.grid, t, self.blend_dist)
            obj.rasterize(self.grid, t, self.blend_dist)

    def apply_velocity(self):
        @ti.kernel
        def clear_mask_velocity():
            for i, j in self.grid.v_grid:
                if self.grid.alpha_grid[i, j] > 0:
                    alpha = self.grid.alpha_grid[i, j]
                    self.grid.vx_grid[i, j] *= 1 - alpha
                    self.grid.vx_grid[i + 1, j] *= 1 - alpha
                    self.grid.vy_grid[i, j] *= 1 - alpha
                    self.grid.vy_grid[i, j + 1] *= 1 - alpha

        @ti.kernel
        def apply_velocity():
            for i, j in self.grid.v_grid:
                if self.grid.alpha_grid[i, j] > 0:
                    alpha = self.grid.alpha_grid[i, j]
                    self.grid.vx_grid[i, j] += self.grid.v_grid[i, j].x * alpha
                    self.grid.vx_grid[i + 1, j] += self.grid.v_grid[i, j].x * alpha
                    self.grid.vy_grid[i, j] += self.grid.v_grid[i, j].y * alpha
                    self.grid.vy_grid[i, j + 1] += self.grid.v_grid[i, j].y * alpha

        clear_mask_velocity()
        apply_velocity()

    def update_velocity(self):
        @ti.kernel
        def update_velocity():
            dx = self.grid.dx
            dt = self.grid.dt
            sx, sy = self.grid.size
            for i, j in ti.ndrange((1, sx), (0, sy)):
                sigma = self.damping_grid[i, j]
                vx = self.grid.vx_grid[i, j]
                self.grid.vx_grid[i, j] -= (
                    self.grid.p_grid[i, j] - self.grid.p_grid[i - 1, j]
                ) / dx * dt + sigma * vx
            for i, j in ti.ndrange((0, sx), (1, sy)):
                sigma = self.damping_grid[i, j]
                vy = self.grid.vy_grid[i, j]
                self.grid.vy_grid[i, j] -= (
                    self.grid.p_grid[i, j] - self.grid.p_grid[i, j - 1]
                ) / dx * dt + sigma * vy

        update_velocity()

    def update_pressure(self):
        @ti.kernel
        def update_pressure():
            dt = self.grid.dt
            c = self.grid.c
            sx, sy = self.grid.size
            for i, j in ti.ndrange((0, sx), (0, sy)):
                alpha = self.grid.alpha_grid[i, j]
                sigma = self.damping_grid[i, j]
                p = self.grid.p_grid[i, j]
                dx = self.grid.vx_grid[i + 1, j] - self.grid.vx_grid[i, j]
                dy = self.grid.vy_grid[i, j + 1] - self.grid.vy_grid[i, j]
                self.grid.p_grid[i, j] -= c**2 * (dx + dy) * dt + sigma * p * (
                    1 - alpha
                )

        update_pressure()

    def step(self):
        # Rasterize geometries.
        self.rasterize(self.t)
        # Update velocity.
        self.update_velocity()
        # Apply boundary velocity.
        self.apply_velocity()
        # Update pressure.
        self.update_pressure()
        self.t += self.grid.dt


def main():
    # Initialize taichi.
    ti.init(arch=ti.gpu)

    parser = argparse.ArgumentParser("object", description="Audio object.")
    parser.add_argument(
        "-o",
        "--open",
        help="Audio wav file.",
        type=pathlib.Path,
    )
    args = parser.parse_args()
    audio_path: pathlib.Path = args.open

    SIZE = (512, 512)
    grid = Grid2D(SIZE, 1 / 512, 340)
    scene = Solver2D(grid)

    # Test objects.
    key_frames = np.array([0, 1], dtype=np.float32)
    center = np.array([[0.5, 0.25], [0.5, 0.75]], dtype=np.float32)
    radius = np.array([0.1, 0.25], dtype=np.float32)
    circle = Circle(key_frames, center, radius)
    # Test audio.
    circle.load_audio_file(audio_path)
    scene.objects.append(circle)

    disp_buf = ti.field(ti.math.vec3, shape=SIZE)

    @ti.kernel
    def clear_disp_buf():
        for i, j in disp_buf:
            disp_buf[i, j] = ti.math.vec3(0)

    @ti.kernel
    def render_alpha():
        """Render pressure field."""
        for i, j in disp_buf:
            disp_buf[i, j] = ti.math.vec3(grid.alpha_grid[i, j])

    @ti.kernel
    def render_velocity():
        """Render pressure field."""
        for i, j in disp_buf:
            x = grid.vx_grid[i, j]
            y = grid.vy_grid[i, j]
            disp_buf[i, j] = ti.math.vec3((x + 1) / 2, (y + 1) / 2, 0)

    gui = ti.GUI("FDTD Simulation", res=SIZE)  # pyright: ignore
    frame = 0
    while gui.running:
        clear_disp_buf()
        scene.rasterize(frame * grid.dt)
        scene.apply_velocity()
        render_velocity()
        gui.set_image(disp_buf)
        gui.text(content=f"frame={frame}, t={frame * grid.dt}", pos=[0, 0])
        gui.show()
        frame += 1


if __name__ == "__main__":
    main()
