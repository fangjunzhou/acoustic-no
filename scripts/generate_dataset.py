from typing import Tuple
import taichi as ti
import argparse
import pathlib
import numpy as np
from scipy.io import wavfile
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm

from acoustic_no.fdtd_solver.grid import Grid2D
from acoustic_no.fdtd_solver.object import Circle
from acoustic_no.fdtd_solver.solver import Solver2D
from acoustic_no.utils.audio_utils import generate_audio_clip


logger = logging.getLogger(__name__)


def main():
    ti.init(arch=ti.gpu)

    parser = argparse.ArgumentParser(
        "fdtd-playground", description="A simple FDTD simulation implemented in Taichi."
    )
    parser.add_argument(
        "-g",
        "--grid",
        help="Simulation grid size.",
        type=lambda s: tuple(map(int, s.split(","))),
        default=(64, 64),
    )
    parser.add_argument(
        "-c", "--cell", help="Simulation cell size.", type=float, default=1 / 64
    )
    parser.add_argument(
        "-b", "--blend", help="Blend distance in unit of dx.", type=float, default=1
    )
    parser.add_argument(
        "-l", "--length", help="Simulation length.", type=float, default=1
    )
    parser.add_argument(
        "-s", "--step-rate", help="Simulation step rate.", type=float, default=44100
    )
    parser.add_argument(
        "-w", "--wave-speed", help="Simulation wave speed.", type=float, default=340
    )
    parser.add_argument(
        "--checkpoint", help="Steps per checkpoint.", type=int, default=1024
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory.",
        type=pathlib.Path,
    )
    args = parser.parse_args()

    # Simulation parameters.
    grid_size: Tuple[int, int] = args.grid
    cell_size: float = args.cell
    blend_dist: float = args.blend
    length: float = args.length
    step_rate: float = args.step_rate
    wave_speed: float = args.wave_speed
    steps_per_checkpoint: int = args.checkpoint
    out_path: pathlib.Path = args.output
    # Create output directory if not exists.
    out_path.mkdir(parents=True, exist_ok=True)

    # Simulation setup.
    step_size = 1 / step_rate
    if step_size > cell_size / (np.sqrt(2) * wave_speed):
        logger.warning(
            f"Step size {step_size} is larger than the CFL condition {cell_size / (np.sqrt(2) * wave_speed)}."
        )
    grid = Grid2D(grid_size, cell_size, step_size, wave_speed)
    solver = Solver2D(grid, 0.0001, 8, 0.05, blend_dist)

    # Scene setup.
    # TODO: Random scene generation.

    # Animated audio source.
    t_frames = np.array([0, length], dtype=np.float32)
    center = np.array(
        [
            [0.25, 0.5],
            [0.75, 0.5],
        ]
    )
    radius = np.array([0.1, 0.1], dtype=np.float32)
    source = Circle(t_frames, center, radius)
    # Generate random audio.
    SAMPLE_RATE = 44100
    NUM_FREQS = 3
    NUM_CLIPS = 25
    clip_length = length / NUM_CLIPS
    t = np.array([])
    sample = np.array([])
    for i in range(NUM_CLIPS):
        freqs = np.random.uniform(20, 2000, size=(NUM_FREQS,))
        amps = np.random.uniform(0, 1, size=(NUM_FREQS,))
        t_clip, sample_clip = generate_audio_clip(
            zip(freqs, amps), SAMPLE_RATE, clip_length
        )
        t_clip += i * clip_length
        t = np.concatenate((t, t_clip))
        sample = np.concatenate((sample, sample_clip))
    # Normalize audio.
    sample = sample / np.max(np.abs(sample))
    sample = 0.25 * sample
    # Save audio.
    audio_path = out_path / "audio.wav"
    wavfile.write(audio_path, SAMPLE_RATE, sample)
    logger.info(f"Source audio saved to {audio_path}")
    # Load audio.
    source.load_audio_sample(SAMPLE_RATE, sample)
    # Plot source normal velocity.
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, 10, source.normal_v.size), source.normal_v)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normal velocity")
    ax.set_title("Normal velocity of the source")
    plt.savefig(out_path / "source_normal_velocity.png")
    plt.close(fig)
    # Add source to scene.
    solver.objects.append(source)

    disp_buf = ti.field(ti.math.vec3, shape=grid.size)

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

    @ti.kernel
    def render_pressure():
        """Render pressure field."""
        for i, j in disp_buf:
            disp_buf[i, j] = ti.math.vec3(grid.p_grid[i, j])

    gui = ti.GUI("FDTD", res=grid.size)  # type: ignore

    # Run solver.
    save_path = out_path / "checkpoints"
    save_path.mkdir(parents=True, exist_ok=True)
    alpha_arr = np.zeros(
        (steps_per_checkpoint, grid.size[0], grid.size[1]), dtype=np.float32
    )
    pressure_arr = np.zeros(
        (steps_per_checkpoint, grid.size[0], grid.size[1]), dtype=np.float32
    )
    velocity_arr = np.zeros(
        (steps_per_checkpoint, grid.size[0], grid.size[1], 2), dtype=np.float32
    )

    num_frames = int(length / grid.dt)
    for frame in tqdm(range(0, num_frames), position=0, leave=True):
        solver.step()
        # Draw pressure field.
        render_pressure()

        # Save alpha and pressure.
        alpha_arr[frame % steps_per_checkpoint] = grid.alpha_grid.to_numpy(
            dtype=np.float32
        )
        pressure_arr[frame % steps_per_checkpoint] = grid.p_grid.to_numpy(
            dtype=np.float32
        )
        velocity_arr[frame % steps_per_checkpoint] = grid.v_grid.to_numpy(
            dtype=np.float32
        ).reshape((grid.size[0], grid.size[1], 2))

        # Save frames.
        if frame % steps_per_checkpoint == steps_per_checkpoint - 1:
            # Save occupancy and pressure.
            np.savez_compressed(
                save_path / f"frame_{frame // steps_per_checkpoint:04d}.npz",
                alpha=alpha_arr,
                pressure=pressure_arr,
                velocity=velocity_arr,
            )
            logger.info(f"Saved frames {frame // steps_per_checkpoint}.")
    # Save unused frames.
    np.savez_compressed(
        save_path / f"frame_{num_frames // steps_per_checkpoint:04d}.npz",
        alpha=alpha_arr[: (num_frames % steps_per_checkpoint)],
        pressure=pressure_arr[: (num_frames % steps_per_checkpoint)],
        velocity=velocity_arr[: (num_frames % steps_per_checkpoint)],
    )
    logger.info(f"Saved frames {num_frames // steps_per_checkpoint}.")


if __name__ == "__main__":
    main()
