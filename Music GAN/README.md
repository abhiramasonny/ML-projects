# Music GAN

This is a transformer GAN that writes MIDI. A Markov chain starts with a melody, and then the generator turns
that rough melody into a full sequence with velocity and timing.

```bash
uv run main.py generate
uv run main.py generate path/to/model.pt
uv run main.py train path/to/midi/folder
```

generate writes .mid files into generated/ training checkpoints go in runs/. The weights aren't
committed, so train once before generate.

the model was trained on MAESTROv3, which is any folder of MIDI you point train at:

```bash
curl -O https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
```