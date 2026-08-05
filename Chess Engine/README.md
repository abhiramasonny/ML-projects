# Chess Engine

this is a bitboard chess engine which can operate from a terminal, or an SDL board.

three files: `chess.h` has the types and declarations, `engine.cpp` has the board,
move generation, evaluation, and search, and `main.cpp` has the UCI, terminal, GUI,
perft, and benchmark modes.

## build

```bash
make
```

it requires SDL2 (`brew install sdl2`).

## play

to play it in the terminal you can use

```bash
./engine
```

type moves in chess notation as `e4`, `Nf3`, `O-O`, or peice to peice moves such as `e2e4`. Other commands: `board`, `fen`, `reset`, `quit`.

for a GUI,

```bash
./engine --gui
```

for a UCI GUI like Arena or ChessBase,

```bash
./engine --uci
```

## testing

perft counts the leaf nodes to a depth and compares against some known values.

```bash
make test
make test-deep
./engine --perft 4
./engine --perft-divide 2 'r3k2r/p1ppqpb1/bn2pnp1/2pPN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1'
./engine --eval 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
```

## benchmarks

`--bench` plays games against Stockfish limited to a range of Elo anchors and fits a maximum
likelihood rating estimate. It needs Stockfish installed (`brew install stockfish`).

```bash
BENCH_GAMES=4 BENCH_MOVETIME_MS=10 BENCH_MAX_PLY=20 BENCH_WORKERS=4 ./engine --bench
```

when I ran the benchmark with the exact command above on my macbook, i got

```
summary_by_elo
vs_sf_elo 1320 score 36/50 pct 0.72 WDL 25/22/3 elo_from_anchor 1480.37
vs_sf_elo 1520 score 27.5/50 pct 0.55 WDL 10/35/5 elo_from_anchor 1554.17
vs_sf_elo 1720 score 26/50 pct 0.52 WDL 8/36/6 elo_from_anchor 1733.63
vs_sf_elo 1920 score 26/50 pct 0.52 WDL 6/40/4 elo_from_anchor 1933.63
vs_sf_elo 2120 score 24/50 pct 0.48 WDL 6/36/8 elo_from_anchor 2106.37
vs_sf_elo 2320 score 24/50 pct 0.48 WDL 5/38/7 elo_from_anchor 2306.37
vs_sf_elo 2520 score 17.5/50 pct 0.35 WDL 0/35/15 elo_from_anchor 2414.7

total_score 181/350
mle_elo 1945
mle_ci95 1892 1998
```

putting the rough ELO of the bot around 1950.

to get a real rating you want an external gauntlet. `gauntlet.py` prints or runs `cutechess-cli`
commands against a list of opponents; edit the constants at the top of the file.

```bash
uv run gauntlet.py --dry-run
uv run gauntlet.py
```