# Stanford CS 231N Acoustic NO

## Setting up the repo

Install package dependencies: 

```
pip install uv
uv sync
source .venv/bin/activate
```

You should see something like this:
```
[Taichi] version 1.7.3, llvm 15.0.7, commit 5ec301be, osx, python 3.12.2
[Taichi] Starting on arch=metal
usage: fdtd-playground [-h] [-g GRID] [-c CELL] [-b BLEND] [-l LENGTH] [-s STEP_RATE] [-w WAVE_SPEED]
                       [--checkpoint CHECKPOINT] [-o OUTPUT]

A simple FDTD simulation implemented in Taichi.

options:
  -h, --help            show this help message and exit
  -g GRID, --grid GRID  Simulation grid size.
  -c CELL, --cell CELL  Simulation cell size.
  -b BLEND, --blend BLEND
                        Blend distance in unit of dx.
  -l LENGTH, --length LENGTH
                        Simulation length.
  -s STEP_RATE, --step-rate STEP_RATE
                        Simulation step rate.
  -w WAVE_SPEED, --wave-speed WAVE_SPEED
                        Simulation wave speed.
  --checkpoint CHECKPOINT
                        Steps per checkpoint.
  -o OUTPUT, --output OUTPUT
                        Output directory.
```

To generate the data:
```
python scripts/generate_dataset.py --help
```
