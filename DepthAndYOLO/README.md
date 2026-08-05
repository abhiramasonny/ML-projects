# Depth + YOLO

this is a live webcam demo running two networks at once. YOLO11 draws boxes around objects, MiDaS
estimates how far away everything is, and each box gets tagged with its depth.

to run,

```bash
uv run main.py
```

You can press q to quit. Both models auto download on first run.