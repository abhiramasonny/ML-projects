# ML Projects

a collection of small AI and simulation projects.

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