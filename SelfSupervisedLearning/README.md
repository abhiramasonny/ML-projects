# Self-Supervised Learning

This is SimCLR on CIFAR-10. it learns image features without a labled dataset.

```bash
uv run main.py train
uv run main.py probe
uv run main.py viz
```

train writes backbone.pth, probe writes probe.pth and prints test accuracy, viz
shows a tSNE of the embeddings. the dataset auto downloads to data on first run.