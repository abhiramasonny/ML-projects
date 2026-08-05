import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ENGINE = "./engine"
CUTECHESS = "cutechess/build/cutechess-cli"
STOCKFISH = "/opt/homebrew/bin/stockfish"
GAMES = 1000
CONCURRENCY = 4
TIME_CONTROL = "10+0.1"
PGN_DIR = "results"
SPRT = None
OPPONENTS = [
    ("stockfish-2200", {"UCI_LimitStrength": "true", "UCI_Elo": 2200, "Hash": 64, "Threads": 1}),
    ("stockfish-2600", {"UCI_LimitStrength": "true", "UCI_Elo": 2600, "Hash": 64, "Threads": 1}),
    ("stockfish-full", {"UCI_LimitStrength": "false", "Hash": 64, "Threads": 1}),
]

dryRun = "--dry-run" in sys.argv

if not dryRun and shutil.which(CUTECHESS) is None:
    sys.exit(f"{CUTECHESS} not found")

runDir = Path(PGN_DIR)
if not dryRun:
    runDir.mkdir(parents=True, exist_ok=True)

pgnPaths = []
for name, options in OPPONENTS:
    pgn = runDir / f"engine_vs_{name}.pgn"
    pgnPaths.append(pgn)

    cmd = [CUTECHESS]
    cmd += ["-engine", "name=engine", f"cmd={Path(ENGINE).resolve()}", "proto=uci",
            "option.Hash=64", "option.Threads=1", "option.Move Overhead=20"]
    cmd += ["-engine", f"name={name}", f"cmd={STOCKFISH}", "proto=uci"]
    cmd += [f"option.{k}={v}" for k, v in sorted(options.items())]
    cmd += ["-each", f"tc={TIME_CONTROL}"]
    cmd += ["-games", str(GAMES), "-repeat"]
    cmd += ["-concurrency", str(CONCURRENCY)]
    cmd += ["-pgnout", str(pgn)]
    if SPRT:
        cmd += ["-sprt", f"elo0={SPRT[0]}", f"elo1={SPRT[1]}", "alpha=0.05", "beta=0.05"]

    print(shlex.join(str(part) for part in cmd))
    if not dryRun and subprocess.run(cmd).returncode != 0:
        sys.exit(1)

if not dryRun and shutil.which("ordo") is not None:
    ordo = ["ordo", "-p", *[str(p) for p in pgnPaths], "-o", str(runDir / "ordo.txt")]
    print(shlex.join(ordo))
    subprocess.run(ordo)
