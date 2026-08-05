# NEAT Snake

This is a NEAT (NeuroEvolution of Augmenting Topologies) algorithm, that plays snake on a 7x7 grid. 
The networks start as a bare input-to-output layer and evolve their own structure, such as growing hidden nodes and connections through mutation.

```bash
uv run main.py train
uv run main.py train --viz
uv run main.py play
```

training saves `pop.pkl` every 10 generations and resumes from it automatically, and `--reset`
starts over.

## inputs

There are 24 inputs. 8 rays around the head, each measuring wall, body, and food distance; 4 global
features for food direction, length, and reachable area; 6 danger/safe flags for the three possible
actions; 3 free space after an action values; 3 safe food after action values; and 3 tail relative
values. 

I messed around for a while and came up with these inputs. even though NEAT generally prefers less inputs, I found that for my training hardware, between 20 - 30 inputs was generally OK.