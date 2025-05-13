from abc import ABC, abstractmethod
import argparse
import logging
import taichi as ti
import pathlib
import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt

from acoustic_no.fdtd_solver.grid import Grid2D

logger = logging.getLogger(__name__)


class Object(ABC):
    @abstractmethod
    def rasterize_alpha(self, scene: Grid2D, t: float, blend_dist: float):
        pass

    @abstractmethod
    def rasterize(self, scene: Grid2D, t: float, blend_dist: float):
        pass


class BoxObstacle(Object):
    key_frames: np.ndarray
    center: np.ndarray
    size: np.ndarray
    rotation: np.ndarray

    def __init__(
        self,
        key_frames: np.ndarray,
        center: np.ndarray,
        size: np.ndarray,
        rotation: np.ndarray,
    ) -> None:
        self.key_frames = key_frames
        self.center = center
        self.size = size
        self.rotation = rotation

        assert key_frames.shape[0] == center.shape[0]
        assert key_frames.shape[0] == size.shape[0]
        assert key_frames.shape[0] == rotation.shape[0]

    def get_param(self, t):
        # Interpolate center and radius.
        center = self.center[-1]
        size = self.size[-1]
        rotation = self.rotation[-1]
        if self.key_frames[-1] <= t:
            return center, size, rotation

        for i in range(self.key_frames.size - 1):
            if self.key_frames[i] <= t:
                curr_t, next_t = self.key_frames[i], self.key_frames[i + 1]
                alpha = (t - curr_t) / (next_t - curr_t)
                curr_c, next_c = self.center[i], self.center[i + 1]
                center = next_c * alpha + curr_c * (1 - alpha)
                curr_s, next_s = self.size[i], self.size[i + 1]
                size = next_s * alpha + curr_s * (1 - alpha)
                curr_r, next_r = self.rotation[i], self.rotation[i + 1]
                rotation = next_r * alpha + curr_r * (1 - alpha)
        return center, size, rotation

    def rasterize_alpha(self, scene: Grid2D, t: float, blend_dist: float):
        @ti.func
        def sdf(p: ti.math.vec2, b: ti.math.vec2):
            d = ti.abs(p) - b
            return ti.math.length(ti.math.max(d, 0)) + ti.math.min(
                ti.math.max(d.x, d.y), 0
            )

        @ti.kernel
        def rasterize_alpha(c: ti.math.vec2, b: ti.math.vec2, r: ti.f32):
            for i, j in scene.alpha_grid:
                pos = ti.math.vec2(i * scene.dx, j * scene.dx)
                pos -= c
                pos = (
                    ti.math.mat2(
                        ti.math.cos(r), -ti.math.sin(r), ti.math.sin(r), ti.math.cos(r)
                    )
                    @ pos
                )
                dist = sdf(pos, b)
                if dist < 0:
                    scene.alpha_grid[i, j] = 1
                # WaveBlender
                elif dist < scene.dx * blend_dist:
                    scene.alpha_grid[i, j] = 1 - dist / (scene.dx * blend_dist)

        c, b, r = self.get_param(t)
        rasterize_alpha(ti.math.vec2(c), ti.math.vec2(b), r.item())

    def rasterize(self, scene: Grid2D, t: float, blend_dist: float):
        @ti.func
        def sdf(p: ti.math.vec2, b: ti.math.vec2):
            d = ti.abs(p) - b
            return ti.math.length(ti.math.max(d, 0)) + ti.math.min(
                ti.math.max(d.x, d.y), 0
            )

        @ti.kernel
        def rasterize_velocity(c: ti.math.vec2, b: ti.math.vec2, r: ti.f32):
            for i, j in scene.v_grid:
                pos = ti.math.vec2(i * scene.dx, j * scene.dx)
                pos -= c
                pos = (
                    ti.math.mat2(
                        ti.math.cos(r), -ti.math.sin(r), ti.math.sin(r), ti.math.cos(r)
                    )
                    @ pos
                )
                dist = sdf(pos, b)
                if dist < scene.dx * blend_dist:
                    scene.v_grid[i, j] = ti.math.vec2(0)

        c, b, r = self.get_param(t)
        rasterize_velocity(ti.math.vec2(c), ti.math.vec2(b), r.item())


class Circle(Object):
    key_frames: np.ndarray
    center: np.ndarray
    radius: np.ndarray
    sample_rate: int = -1
    samples: np.ndarray = np.array([])
    normal_v: np.ndarray = np.array([])

    def __init__(
        self, key_frames: np.ndarray, center: np.ndarray, radius: np.ndarray
    ) -> None:
        self.key_frames = key_frames
        self.center = center
        self.radius = radius

    def load_audio_sample(self, sample_rate: int, samples: np.ndarray):
        self.sample_rate = sample_rate
        self.samples = samples
        self.integrate_velocity()

    def load_audio_file(self, audio_path: pathlib.Path):
        self.sample_rate, self.samples = read(audio_path)
        self.integrate_velocity()

    def integrate_velocity(self, drift_correction: int = 256, avg_sample: int = 128):
        # Numerical integration.
        self.normal_v = np.cumsum(self.samples / self.sample_rate)
        # Solve drift.
        trim = self.normal_v.size % avg_sample
        trim_normal_v = self.normal_v
        if trim != 0:
            trim_normal_v = self.normal_v[:-trim]
        avg = trim_normal_v.reshape((-1, avg_sample))
        avg = np.mean(avg, axis=-1).flatten()
        low_res_idx = np.arange(0, avg.size - 1, int(avg.size / drift_correction))
        low_res = avg[low_res_idx]
        offset = np.interp(
            np.linspace(0, low_res.size, self.normal_v.size),
            np.arange(0, low_res.size),
            low_res,
        )
        self.normal_v -= offset
        self.normal_v /= np.max(np.abs(self.normal_v))

    def get_center_radius(self, t):
        # Interpolate center and radius.
        center = self.center[-1]
        radius = self.radius[-1]
        if self.key_frames[-1] <= t:
            return center, radius

        for i in range(self.key_frames.size - 1):
            if self.key_frames[i] <= t:
                curr_t, next_t = self.key_frames[i], self.key_frames[i + 1]
                alpha = (t - curr_t) / (next_t - curr_t)
                curr_c, next_c = self.center[i], self.center[i + 1]
                center = next_c * alpha + curr_c * (1 - alpha)
                curr_r, next_r = self.radius[i], self.radius[i + 1]
                radius = next_r * alpha + curr_r * (1 - alpha)
        return center, radius

    def get_normal_v(self, t):
        velocity = np.array([0])
        i = int(self.sample_rate * t)
        if i >= 0 and i < self.samples.size - 1:
            alpha = self.sample_rate * t - i
            curr_v, next_v = self.normal_v[i], self.normal_v[i + 1]
            velocity = curr_v * alpha + next_v * (1 - alpha)
        return velocity

    def rasterize_alpha(self, scene: Grid2D, t: float, blend_dist: float):
        c, r = self.get_center_radius(t)

        @ti.kernel
        def rasterize_alpha(c: ti.math.vec2, r: ti.f32):
            for i, j in scene.alpha_grid:
                pos = ti.math.vec2(i * scene.dx, j * scene.dx)
                dist = ti.math.length(pos - c)
                if dist < r:
                    scene.alpha_grid[i, j] = 1
                # WaveBlender
                elif dist < r + scene.dx * blend_dist:
                    scene.alpha_grid[i, j] = 1 - (dist - r) / (scene.dx * blend_dist)

        rasterize_alpha(ti.math.vec2(c), r.item())

    def rasterize(self, scene: Grid2D, t: float, blend_dist: float):
        @ti.kernel
        def rasterize_velocity(c: ti.math.vec2, r: ti.f32, v: ti.f32):
            for i, j in scene.v_grid:
                pos = ti.math.vec2(i * scene.dx, j * scene.dx)
                dist = ti.math.length(pos - c)
                if dist < r + scene.dx * blend_dist:
                    normal = (pos - c) / dist
                    scene.v_grid[i, j] = v * normal

        c, r = self.get_center_radius(t)
        v = self.get_normal_v(t)

        rasterize_velocity(ti.math.vec2(c), r.item(), v.item())


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

    # Test objects.
    key_frames = np.array([0, 1], dtype=np.float32)
    center = np.array([[0.5, 0.25], [0.5, 0.75]], dtype=np.float32)
    radius = np.array([0.25, 0.25], dtype=np.float32)
    circle = Circle(key_frames, center, radius)
    # Test audio.
    sample_rate = 44100
    audio_length = 8
    samples_t = np.linspace(0, audio_length, int(sample_rate * audio_length))
    samples = np.sin(440 * 2 * np.pi * samples_t)
    circle.load_audio_file(audio_path)
    # circle.load_audio_sample(sample_rate, samples)
    num_samples = circle.samples.size
    audio_length = num_samples / circle.sample_rate
    samples_t = np.linspace(0, audio_length, num_samples)
    plt.plot(samples_t, circle.normal_v)
    plt.show()


if __name__ == "__main__":
    main()
