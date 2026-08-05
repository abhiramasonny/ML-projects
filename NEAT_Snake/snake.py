import random
from collections import deque

GRID = 7
NIN, NOUT = 43, 3
MAX_STEPS = 700
PLANNER_EVERY = 4
PLANNER_DEPTH = 30
PLANNER_NODES = 160

UP, DN, LT, RT = 0, 1, 2, 3
fwd = {UP: (0, -1), RT: (1, 0), DN: (0, 1), LT: (-1, 0)}
rgt = {UP: (1, 0), RT: (0, 1), DN: (-1, 0), LT: (0, -1)}
turnLeft = {UP: LT, LT: DN, DN: RT, RT: UP}
turnRight = {UP: RT, RT: DN, DN: LT, LT: UP}
RAYS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


class Snake:
    def __init__(self, rng=None):
        self.rng = rng or random
        self.reset()

    def reset(self):
        c = GRID // 2
        self.body = [(c, c), (c - 1, c), (c - 2, c)]
        self.dir = RT
        self.steps = 0
        self.apples = 0
        self.maxSteps = MAX_STEPS
        self.food = None
        self.place()
        self.recentHeads = deque(maxlen=GRID * GRID)
        self.stateCounts = {}
        self.clearCaches()
        self.rememberState()

    def clearCaches(self):
        self.bodyCache = {}
        self.survivalCache = {}
        self.spaceCache = {}
        self.safeFoodCache = {}
        self.planCache = {}
        self.loopCache = None

    def place(self):
        taken = set(self.body)
        free = [(x, y) for x in range(GRID) for y in range(GRID) if (x, y) not in taken]
        self.food = self.rng.choice(free) if free else None

    def stateKey(self):
        return (self.body[0], self.dir, self.food, tuple(self.body[:min(8, len(self.body))]))

    def rememberState(self):
        self.recentHeads.append(self.body[0])
        key = self.stateKey()
        self.stateCounts[key] = self.stateCounts.get(key, 0) + 1
        self.loopCache = None

    def loopPressure(self):
        if self.loopCache is not None:
            return self.loopCache
        if len(self.recentHeads) < 8:
            self.loopCache = 0.0
            return 0.0
        recent = list(self.recentHeads)
        repeatRatio = 1.0 - len(set(recent)) / max(1, len(recent))
        exactRepeats = self.stateCounts.get(self.stateKey(), 0)
        self.loopCache = max(repeatRatio, min(1.0, exactRepeats / 3.0))
        return self.loopCache

    def recentHeadPenalty(self, cell):
        recent = list(self.recentHeads)
        if not recent:
            return 0.0
        count = 0.0
        ageBonus = 0.0
        for i, c in enumerate(recent):
            if c == cell:
                count += 1.0
                ageBonus += (i + 1) / len(recent)
        return count + ageBonus

    def dirAfterTurn(self, turn):
        if turn == 0:
            return turnLeft[self.dir]
        if turn == 2:
            return turnRight[self.dir]
        return self.dir

    def bodyAfterAction(self, turn):
        if turn in self.bodyCache:
            return self.bodyCache[turn]

        hx, hy = self.body[0]
        dx, dy = fwd[self.dirAfterTurn(turn)]
        nx, ny = hx + dx, hy + dy
        head = (nx, ny)

        if nx < 0 or nx >= GRID or ny < 0 or ny >= GRID:
            self.bodyCache[turn] = (None, None, False)
            return self.bodyCache[turn]

        ate = self.food is not None and head == self.food
        if head in (self.body[1:] if ate else self.body[1:-1]):
            self.bodyCache[turn] = (None, None, False)
            return self.bodyCache[turn]

        newBody = [head] + (self.body if ate else self.body[:-1])
        self.bodyCache[turn] = (head, newBody, ate)
        return self.bodyCache[turn]

    def bodyAfterCell(self, body, cell, food):
        x, y = cell
        if x < 0 or x >= GRID or y < 0 or y >= GRID:
            return None, False
        ate = food is not None and cell == food
        if cell in (body[1:] if ate else body[1:-1]):
            return None, False
        return [cell] + list(body if ate else body[:-1]), ate

    def floodCount(self, start, obstacles):
        if start is None:
            return 0
        seen = set()
        stack = [start]
        while stack:
            c = stack.pop()
            x, y = c
            if c in seen or c in obstacles or x < 0 or x >= GRID or y < 0 or y >= GRID:
                continue
            seen.add(c)
            stack += [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return len(seen)

    def pathExists(self, start, target, obstacles):
        if start is None or target is None:
            return False
        if start == target:
            return True
        seen = set()
        stack = [start]
        while stack:
            c = stack.pop()
            x, y = c
            if c == target:
                return True
            if c in seen or c in obstacles or x < 0 or x >= GRID or y < 0 or y >= GRID:
                continue
            seen.add(c)
            stack += [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return False

    def shortestPath(self, start, target, obstacles):
        if start is None or target is None:
            return None
        q = deque([start])
        parent = {start: None}
        while q:
            c = q.popleft()
            if c == target:
                path = []
                while c != start:
                    path.append(c)
                    c = parent[c]
                path.reverse()
                return path
            x, y = c
            for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                nx, ny = nb
                if nx < 0 or nx >= GRID or ny < 0 or ny >= GRID or nb in obstacles or nb in parent:
                    continue
                parent[nb] = c
                q.append(nb)
        return None

    def bfsDistance(self, start, target, obstacles):
        if start is None or target is None:
            return None
        if start == target:
            return 0
        q = deque([(start, 0)])
        seen = {start}
        while q:
            c, d = q.popleft()
            x, y = c
            for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                nx, ny = nb
                if nx < 0 or nx >= GRID or ny < 0 or ny >= GRID or nb in obstacles or nb in seen:
                    continue
                if nb == target:
                    return d + 1
                seen.add(nb)
                q.append((nb, d + 1))
        return None

    def survivalScore(self, body):
        tailOk = self.pathExists(body[0], body[-1], set(body[1:-1]))
        free = self.floodCount(body[0], set(body[1:]))
        return tailOk, free, free - len(body)

    def tailDistance(self, body):
        d = self.bfsDistance(body[0], body[-1], set(body[1:-1]))
        return -999.0 if d is None else float(d)

    def foodDistance(self, body):
        if self.food is None:
            return 0.0
        d = self.bfsDistance(body[0], self.food, set(body[1:-1]))
        return 999.0 if d is None else float(d)

    def survivableAfter(self, turn):
        if turn in self.survivalCache:
            return self.survivalCache[turn]
        _, newBody, _ = self.bodyAfterAction(turn)
        if newBody is None:
            self.survivalCache[turn] = -1.0
        else:
            tailOk, free, margin = self.survivalScore(newBody)
            self.survivalCache[turn] = 1.0 if tailOk and margin >= 0 else (0.25 if tailOk else -1.0)
        return self.survivalCache[turn]

    def spaceAfter(self, turn):
        if turn in self.spaceCache:
            return self.spaceCache[turn]
        head, newBody, _ = self.bodyAfterAction(turn)
        if newBody is None:
            self.spaceCache[turn] = 0.0
        else:
            self.spaceCache[turn] = self.floodCount(head, set(newBody[1:])) / (GRID * GRID)
        return self.spaceCache[turn]

    def safeFoodAfter(self, turn):
        if turn in self.safeFoodCache:
            return self.safeFoodCache[turn]

        head, newBody, ate = self.bodyAfterAction(turn)
        if newBody is None or self.food is None:
            self.safeFoodCache[turn] = -1.0
            return -1.0

        if ate:
            tailOk, free, margin = self.survivalScore(newBody)
            self.safeFoodCache[turn] = 1.0 if tailOk and margin >= 0 else -1.0
            return self.safeFoodCache[turn]

        path = self.shortestPath(head, self.food, set(newBody[1:-1]))
        if path is None:
            self.safeFoodCache[turn] = -1.0
            return -1.0

        simBody = newBody
        gotFood = False
        for cell in path:
            simBody, gotFood = self.bodyAfterCell(simBody, cell, self.food)
            if simBody is None:
                self.safeFoodCache[turn] = -1.0
                return -1.0

        if not gotFood:
            self.safeFoodCache[turn] = -1.0
            return -1.0

        tailOk, free, margin = self.survivalScore(simBody)
        self.safeFoodCache[turn] = 1.0 if tailOk and margin >= 0 else (0.25 if tailOk else -1.0)
        return self.safeFoodCache[turn]

    def escapeScore(self, turn, safeFoodBest):
        _, newBody, ate = self.bodyAfterAction(turn)
        if newBody is None:
            return -1e9

        tailOk, free, margin = self.survivalScore(newBody)
        if not tailOk:
            return -1e8

        score = 18.0 * free + 30.0 * margin + 7.0 * self.tailDistance(newBody)
        score -= 35.0 * self.recentHeadPenalty(newBody[0])
        foodDist = self.foodDistance(newBody)
        score += -2.5 * foodDist if safeFoodBest > 0.0 else 1.5 * foodDist

        if ate:
            score += 80.0 if margin >= 0 else -500.0
        return score

    def advanceSim(self, body, direction, action, food):
        if action == 0:
            ndir = turnLeft[direction]
        elif action == 2:
            ndir = turnRight[direction]
        else:
            ndir = direction

        hx, hy = body[0]
        dx, dy = fwd[ndir]
        nx, ny = hx + dx, hy + dy
        if nx < 0 or nx >= GRID or ny < 0 or ny >= GRID:
            return None, None, False

        head = (nx, ny)
        ate = food is not None and head == food
        if head in (body[1:] if ate else body[1:-1]):
            return None, None, False
        return ndir, (head,) + tuple(body if ate else body[:-1]), ate

    def plan(self):
        if self.food is None:
            return None

        key = (tuple(self.body), self.dir, self.food)
        if key in self.planCache:
            return self.planCache[key]

        startBody = tuple(self.body)
        q = deque([(startBody, self.dir, None, 0)])
        seen = {(startBody, self.dir)}
        best = None
        expanded = 0

        while q and expanded < PLANNER_NODES:
            body, direction, firstAction, depth = q.popleft()
            expanded += 1
            if depth >= PLANNER_DEPTH:
                continue

            for action in (0, 1, 2):
                ndir, newBody, ate = self.advanceSim(body, direction, action, self.food)
                if newBody is None:
                    continue
                actualFirst = action if firstAction is None else firstAction

                if ate:
                    tailOk, free, margin = self.survivalScore(list(newBody))
                    if tailOk and margin >= 0:
                        score = 10000.0 + 100.0 * margin + 25.0 * free - 20.0 * depth
                        if best is None or score > best["score"]:
                            best = {"action": actualFirst, "score": score, "free": free, "margin": margin}
                    continue

                if (newBody, ndir) in seen:
                    continue
                seen.add((newBody, ndir))
                q.append((newBody, ndir, actualFirst, depth + 1))

        self.planCache[key] = best
        return best

    def reach(self):
        hx, hy = self.body[0]
        obstacles = set(self.body)
        seen = set()
        stack = [(hx + 1, hy), (hx - 1, hy), (hx, hy + 1), (hx, hy - 1)]
        while stack:
            c = stack.pop()
            x, y = c
            if c in seen or c in obstacles or x < 0 or x >= GRID or y < 0 or y >= GRID:
                continue
            seen.add(c)
            stack += [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return len(seen)

    def inputs(self):
        hx, hy = self.body[0]
        bodySet = set(self.body[1:])
        fx, fy = fwd[self.dir]
        rx, ry = rgt[self.dir]
        feats = []

        for a, b in RAYS:
            dx, dy = a * fx + b * rx, a * fy + b * ry
            wall = body = food = 0.0
            x, y, dist = hx, hy, 0
            while True:
                x += dx
                y += dy
                dist += 1
                if x < 0 or x >= GRID or y < 0 or y >= GRID:
                    wall = 1.0 / dist
                    break
                if body == 0.0 and (x, y) in bodySet:
                    body = 1.0 / dist
                if self.food is not None and (x, y) == self.food:
                    food = 1.0
            feats += [wall, body, food]

        gx, gy = self.food if self.food is not None else (hx, hy)
        dvx, dvy = gx - hx, gy - hy
        feats += [
            (dvx * fx + dvy * fy) / GRID,
            (dvx * rx + dvy * ry) / GRID,
            len(self.body) / (GRID * GRID),
            self.reach() / (GRID * GRID),
        ]

        for turn in (0, 1, 2):
            blocked = self.bodyAfterAction(turn)[1] is None
            feats += [1.0, 0.0] if blocked else [0.0, 1.0]
        for turn in (0, 1, 2):
            feats.append(self.spaceAfter(turn))
        for turn in (0, 1, 2):
            feats.append(self.safeFoodAfter(turn))

        tx, ty = self.body[-1]
        dvx, dvy = tx - hx, ty - hy
        feats += [
            (dvx * fx + dvy * fy) / GRID,
            (dvx * rx + dvy * ry) / GRID,
            (abs(dvx) + abs(dvy)) / (2 * (GRID - 1)),
        ]
        return feats

    def step(self, a):
        if a == 0:
            self.dir = turnLeft[self.dir]
        elif a == 2:
            self.dir = turnRight[self.dir]

        hx, hy = self.body[0]
        dx, dy = fwd[self.dir]
        nx, ny = hx + dx, hy + dy
        if nx < 0 or nx >= GRID or ny < 0 or ny >= GRID:
            return False

        head = (nx, ny)
        ate = self.food is not None and head == self.food
        if head in (self.body[1:] if ate else self.body[1:-1]):
            return False

        self.body.insert(0, head)
        self.steps += 1
        if ate:
            self.apples += 1
            self.maxSteps = MAX_STEPS + self.apples * 50
            self.place()
        else:
            self.body.pop()

        self.clearCaches()
        self.rememberState()
        return self.steps < self.maxSteps


def chooseAction(g, s):
    out = g.activate(s.inputs())
    legal = [a for a in (0, 1, 2) if s.bodyAfterAction(a)[1] is not None]
    if not legal:
        return out.index(max(out))

    survivable = [a for a in legal if s.survivableAfter(a) > 0.0]
    candidates = survivable or legal
    loopPressure = s.loopPressure()
    safeVals = {a: s.safeFoodAfter(a) for a in candidates}
    safeFoodActions = [a for a in candidates if safeVals[a] > 0.0]

    if safeFoodActions and loopPressure < 0.65:
        def foodScore(a):
            _, newBody, ate = s.bodyAfterAction(a)
            tailOk, free, margin = s.survivalScore(newBody)
            score = 12.0 * out[a] - 8.0 * s.foodDistance(newBody) + 4.0 * free + 10.0 * margin
            if ate:
                score += 120.0
            if not tailOk:
                score -= 1000.0
            return score
        return max(safeFoodActions, key=foodScore)

    plan = None
    if s.steps % PLANNER_EVERY == 0 or loopPressure > 0.55 or not safeFoodActions:
        plan = s.plan()

    if plan is not None and plan["action"] in candidates:
        action = plan["action"]
        if plan["margin"] >= 0 and plan["free"] >= len(s.body):
            return action
        if out[action] >= max(out[a] for a in candidates) - 0.25:
            return action

    safeFoodBest = max(safeVals.values()) if safeVals else -1.0

    def escape(a):
        head = s.bodyAfterAction(a)[0]
        score = s.escapeScore(a, safeFoodBest) + 3.0 * out[a]
        return score - 25.0 * loopPressure * s.recentHeadPenalty(head)

    return max(candidates, key=escape)

