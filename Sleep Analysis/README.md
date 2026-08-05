# Sleep Analysis

This is automatic sleep staging from a single EEG channel. it reads one night of raw brain signal and
labels every 30 seconds as wake, N1, N2, N3, or REM.

```bash
uv run main.py download
uv run main.py preprocess
uv run main.py train
uv run main.py
```

If you pass in no argument it auto runs all three.

I used Sleep-EDF from PhysioNet, downloaded by MNE. Each subject has a PSG recording and a hypnogram
file where an expert marked every 30-second window. 