# AI Projects

a collection of small AI and simulation projects. most are a single file; the two big ones
are split into three. they're meant to be read as much as run — no framework, no config
system, no `src/` folder to dig through. open `main.py` and read it top to bottom.

## setup

everything shares one environment.

```bash
uv sync
```

then run any project from its own folder:

```bash
cd "Stable Fluids" && uv run main.py
cd NEAT_Snake && uv run main.py play
```

the chess engine is C++ and builds with `make` instead.

## the projects

| project | what it is |
|---------|-----------|
| [Chess Engine](Chess%20Engine) | bitboard engine in one C++ file — UCI, terminal, and SDL board |
| [NEAT Snake](NEAT_Snake) | networks that evolve their own topology to play snake |
| [PPO Snake](PPO_Snake) | the same game solved with reinforcement learning instead |
| [Stable Fluids](Stable%20Fluids) | real-time fluid sim you paint with the mouse |
| [NavierStokes PINN](NavierStokes%20PINN) | five classic flows solved by neural nets with no mesh |
| [Triple Pendulum PINN](TripplePendulumPINN) | learning a chaotic system, checked against RK45 |
| [Self-Supervised Learning](SelfSupervisedLearning) | SimCLR learning image features with zero labels |
| [Sleep Analysis](Sleep%20Analysis) | staging a night of sleep from one EEG channel |
| [Handwriting OCR](HandwritingOCR) | draw a math symbol, it names it |
| [Music GAN](Music%20GAN) | a transformer GAN that writes MIDI |
| [Depth + YOLO](DepthAndYOLO) | webcam demo tagging detected objects with distance |

each folder has its own README explaining the idea and the parts worth knowing about.

## conventions

- one `main.py` per project, constants at the top, no argument parsing
- if a project outgrows one file it splits into two or three, never a `src/` tree
- run modes are a bare word: `uv run main.py train`, `uv run main.py play`
- no comments — if something needs explaining it goes in the README
- datasets and model outputs are gitignored
# ML-projects
