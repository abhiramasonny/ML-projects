# PPO Snake

This Snake played with reinforcement learning (PPO) on a 7x7 grid. The snake starts completely randomly and learns
from reward by playing more and more games.

```bash
uv run main.py train
uv run main.py train --viz
uv run main.py play
```

Training resumes from policy.pth automatically and saves every 3 updates. --viz shows a
live view of the network weights and reward charts for a cool thing to see while its training.

## Reward

+1 for eating food, -1 for dying, and +-0.01 for moving toward or away from food.

That small shaping reward helps learning early on. There is also a hunger limit so the snake
cannot survive forever by looping around. theres prolly better rewards out there but this relativly simple system led to faster training and produced a decent result (clears a 6x6 grid all the time and nearly finishes a 7x7 grid with under 6 hours of training).
