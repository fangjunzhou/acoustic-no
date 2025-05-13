from typing import Iterator, Tuple
import numpy as np


def generate_audio_clip(
    freq_profiles: Iterator[Tuple[float, float]], sample_rate: int, length: float
):
    """Generate an audio clip with a given frequency profile.

    :param freq_profiles: frequency profile of the audio clip, a list of tuple of (frequency, amplitude)
    :param sample_rate: sample rate of the audio clip
    :param length: length of the audio clip in seconds
    """
    # Time vector
    t = np.linspace(0, length, int(length * sample_rate), endpoint=False)
    # Sample the audio clip
    sample = np.zeros_like(t)
    for freq, amp in freq_profiles:
        sample += amp * np.sin(2 * np.pi * freq * t)
    return t, sample
