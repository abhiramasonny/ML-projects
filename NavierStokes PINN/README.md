# NavierStokes PINN

These are PINNs that solve the 2D incompressible Navier-Stokes equations. Instead of using a mesh,
a small MLP maps (x,y) to velocity and pressure, and the loss is the PDE residual itself,
computed with autograd. No training data is used.

```bash
uv run main.py cavity
uv run main.py kovasznay
uv run main.py taylorgreen
uv run main.py poiseuille
uv run main.py couette
```

Each case writes field plots, a loss curve, a metrics.json file, and the trained model to outputs.

## accuracy

| case | L2 |
|------|-------------|
| couette | 0.019% |
| poiseuille | 0.015% |
| kovasznay Re 40 | 0.017% |
| taylor-green, mean over time | 0.34% |
| cavity Re 100, u centerline | ~10% |
| cavity Re 400 | ~36% |
| cavity Re 1000 | ~59% |

we could prolly get it higher with more training.