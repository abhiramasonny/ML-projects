# Handwriting OCR

You can draw any symbol with the mouse and a CNN tells you what it is. It recognizes 369
symbols including letters, greek symbols, numbers, arrows, etc.

```bash
uv run main.py prep
uv run main.py train
uv run main.py
```

prep downloads HASYv2 (dataset) and builds a cache, train actually trains the model, and running it wouthout an
argument opens the drawing GUI.

## training
The dataset is very imbalanced across 369 classes, so the training uses random affine
augmentation to multiply the ones that are disproportionally featured, plus label smoothing.
