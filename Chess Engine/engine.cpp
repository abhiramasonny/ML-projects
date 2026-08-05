#include "chess.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>

const int INF = 1000000;
const int MATE = 900000;
const int PIECE_VALUE[6] = {100, 320, 330, 500, 900, 20000};

const int PST[6][64] = {
  {  0,  0,  0,  0,  0,  0,  0,  0,
     5,  5,  5,  0,  0,  5,  5,  5,
     8, 10, 12, 18, 18, 12, 10,  8,
    10, 12, 18, 28, 28, 18, 12, 10,
    14, 16, 22, 35, 35, 22, 16, 14,
    20, 24, 30, 45, 45, 30, 24, 20,
    45, 50, 55, 65, 65, 55, 50, 45,
     0,  0,  0,  0,  0,  0,  0,  0 },
  {-50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50 },
  {-20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20 },
  {  0,  0,  5, 10, 10,  5,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  5,  5,  0,  0,  0 },
  {-20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -10,  5,  5,  5,  5,  5,  0,-10,
     0,  0,  5,  5,  5,  5,  0, -5,
    -5,  0,  5,  5,  5,  5,  0, -5,
   -10,  0,  5,  5,  5,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20 },
  { 20, 30, 10,  0,  0, 10, 30, 20,
    20, 20,  0,  0,  0,  0, 20, 20,
   -10,-20,-20,-20,-20,-20,-20,-10,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30 }
};

const int KNIGHT_DIRS[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
const int KING_DIRS[8][2] = {{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1},{0,-1},{1,-1}};
const int BISHOP_DIRS[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
const int ROOK_DIRS[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
const int QUEEN_DIRS[8][2] = {{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};

static u64 zobPiece[64][12];
static u64 zobSide;
static u64 zobCastle[16];
static u64 zobEp[8];

void initZobrist() {
  u64 seed = 0x9E3779B97F4A7C15ULL;
  auto next = [&]() { seed ^= seed >> 33; seed *= 0xBF58476D1CE4E5B9ULL; return seed; };
  for (int i = 0; i < 64; i++)
    for (int j = 0; j < 12; j++) zobPiece[i][j] = next();
  zobSide = next();
  for (int i = 0; i < 16; i++) zobCastle[i] = next();
  for (int i = 0; i < 8; i++) zobEp[i] = next();
}

u64 calcHash(const Board& b) {
  u64 h = 0;
  for (int c = WHITE; c <= BLACK; ++c) {
    for (int p = P; p <= K; ++p) {
      u64 pieces = b.bb[c][p];
      while (pieces) {
        int sq = __builtin_ctzll(pieces);
        pieces &= pieces - 1;
        h ^= zobPiece[sq][c * 6 + p];
      }
    }
  }
  if (b.stm == BLACK) h ^= zobSide;
  h ^= zobCastle[b.castle & 15];
  if (b.ep != A1) h ^= zobEp[b.ep % 8];
  return h;
}

void copyPosition(const Board& src, Board& dst) {
  memcpy(dst.bb, src.bb, sizeof(dst.bb));
  memcpy(dst.occ, src.occ, sizeof(dst.occ));
  dst.stm = src.stm;
  dst.ply = src.ply;
  dst.zh = src.zh;
  dst.castle = src.castle;
  dst.ep = src.ep;
  dst.halfmove = src.halfmove;
  dst.fullmove = src.fullmove;
}

void Board::fromFen(const std::string& fen) {
  memset(bb, 0, sizeof(bb));
  memset(occ, 0, sizeof(occ));
  zh = 0;
  ply = 0;
  stm = WHITE;
  castle = 0;
  ep = A1;
  halfmove = 0;
  fullmove = 1;

  std::istringstream parts(fen);
  std::string placement, active, rights, eps, halfmoveStr, fullmoveStr;
  parts >> placement >> active >> rights >> eps >> halfmoveStr >> fullmoveStr;

  int sq = 56;
  for (size_t i = 0; i < placement.size(); i++) {
    char ch = placement[i];
    if (ch == '/') { sq -= 16; continue; }
    if (isdigit(ch)) { sq += ch - '0'; continue; }
    Co c = isupper(ch) ? WHITE : BLACK;
    char pc = tolower(ch);
    Pc p = pc == 'p' ? P : pc == 'n' ? N : pc == 'b' ? B : pc == 'r' ? R : pc == 'q' ? Q : K;
    bb[c][p] |= 1ULL << sq;
    sq++;
  }

  for (int c = WHITE; c <= BLACK; ++c)
    for (int p = P; p <= K; ++p) occ[c] |= bb[c][p];

  if (active == "b") stm = BLACK;
  if (rights.find('K') != std::string::npos) castle |= 1;
  if (rights.find('Q') != std::string::npos) castle |= 2;
  if (rights.find('k') != std::string::npos) castle |= 4;
  if (rights.find('q') != std::string::npos) castle |= 8;
  if (eps.size() == 2 && eps != "-") ep = (Sq)((eps[0] - 'a') + (eps[1] - '1') * 8);
  if (!halfmoveStr.empty()) halfmove = (u32)atoi(halfmoveStr.c_str());
  if (!fullmoveStr.empty()) fullmove = (u32)atoi(fullmoveStr.c_str());

  zh = calcHash(*this);
  hist[0].zh = zh;
}

std::string Board::toFen() const {
  std::string fen;
  for (int r = 7; r >= 0; r--) {
    int empty = 0;
    for (int f = 0; f < 8; f++) {
      u64 mask = 1ULL << (r * 8 + f);
      char found = 0;
      for (int c = WHITE; c <= BLACK && !found; ++c)
        for (int p = P; p <= K; ++p)
          if (bb[c][p] & mask) {
            found = c == BLACK ? tolower("PNBRQK"[p]) : "PNBRQK"[p];
            break;
          }
      if (found) {
        if (empty) { fen += (char)('0' + empty); empty = 0; }
        fen += found;
      } else {
        empty++;
      }
    }
    if (empty) fen += (char)('0' + empty);
    if (r > 0) fen += '/';
  }
  fen += ' ';
  fen += stm == WHITE ? 'w' : 'b';
  fen += ' ';
  if (!castle) fen += '-';
  if (castle & 1) fen += 'K';
  if (castle & 2) fen += 'Q';
  if (castle & 4) fen += 'k';
  if (castle & 8) fen += 'q';
  fen += ' ';
  if (ep == A1) fen += '-';
  else { fen += char('a' + ep % 8); fen += char('1' + ep / 8); }
  fen += " " + std::to_string(halfmove) + " " + std::to_string(fullmove);
  return fen;
}

void Board::doMove(Move m) {
  hist[ply] = {zh, halfmove, fullmove, castle, ep, 255, P};
  zh ^= zobCastle[castle & 15];
  if (ep != A1) zh ^= zobEp[ep % 8];

  u64 from = 1ULL << m.from();
  u64 to = 1ULL << m.to();
  Pc p = K;
  for (int i = P; i <= K; ++i)
    if (bb[stm][i] & from) { p = (Pc)i; break; }

  bb[stm][p] &= ~from;
  bb[stm][p] |= to;
  zh ^= zobPiece[m.from()][stm * 6 + p] ^ zobPiece[m.to()][stm * 6 + p];

  bool capture = false;
  for (int i = P; i <= K; ++i) {
    if (bb[1 - stm][i] & to) {
      bb[1 - stm][i] &= ~to;
      zh ^= zobPiece[m.to()][(1 - stm) * 6 + i];
      capture = true;
      hist[ply].capSq = (u8)m.to();
      hist[ply].capPc = (Pc)i;
    }
  }

  if (p == P && ep != A1 && m.to() == ep && !(to & (occ[WHITE] | occ[BLACK]))) {
    Sq capsq = stm == WHITE ? (Sq)(m.to() - 8) : (Sq)(m.to() + 8);
    bb[1 - stm][P] &= ~(1ULL << capsq);
    zh ^= zobPiece[capsq][(1 - stm) * 6 + P];
    capture = true;
    hist[ply].capSq = (u8)capsq;
    hist[ply].capPc = P;
  }

  if (m.prom()) {
    bb[stm][P] &= ~to;
    bb[stm][m.prom()] |= to;
    zh ^= zobPiece[m.to()][stm * 6 + P] ^ zobPiece[m.to()][stm * 6 + m.prom()];
  }

  if (p == K) {
    if (stm == WHITE) castle &= ~(1 | 2);
    else castle &= ~(4 | 8);
    Sq rookFrom = A1, rookTo = A1;
    if (m.from() == E1 && m.to() == G1) { rookFrom = H1; rookTo = F1; }
    else if (m.from() == E1 && m.to() == C1) { rookFrom = A1; rookTo = D1; }
    else if (m.from() == E8 && m.to() == G8) { rookFrom = H8; rookTo = F8; }
    else if (m.from() == E8 && m.to() == C8) { rookFrom = A8; rookTo = D8; }
    if (rookTo != A1) {
      bb[stm][R] &= ~(1ULL << rookFrom);
      bb[stm][R] |= 1ULL << rookTo;
      zh ^= zobPiece[rookFrom][stm * 6 + R] ^ zobPiece[rookTo][stm * 6 + R];
    }
  }

  if (p == R) {
    if (m.from() == H1) castle &= ~1;
    if (m.from() == A1) castle &= ~2;
    if (m.from() == H8) castle &= ~4;
    if (m.from() == A8) castle &= ~8;
  }
  if (capture) {
    if (m.to() == H1) castle &= ~1;
    if (m.to() == A1) castle &= ~2;
    if (m.to() == H8) castle &= ~4;
    if (m.to() == A8) castle &= ~8;
  }

  ep = A1;
  if (p == P && abs((int)m.to() - (int)m.from()) == 16) ep = (Sq)((m.from() + m.to()) / 2);

  if (p == P || capture) halfmove = 0;
  else halfmove++;

  for (int c = WHITE; c <= BLACK; ++c) {
    occ[c] = 0;
    for (int i = P; i <= K; ++i) occ[c] |= bb[c][i];
  }

  if (stm == BLACK) fullmove++;
  stm = (Co)(1 - stm);
  ply++;
  zh ^= zobCastle[castle & 15];
  if (ep != A1) zh ^= zobEp[ep % 8];
  zh ^= zobSide;
}

void Board::undoMove(Move m, u64 zho) {
  stm = (Co)(1 - stm);
  ply--;
  castle = hist[ply].castle;
  ep = hist[ply].ep;
  halfmove = hist[ply].halfmove;
  fullmove = hist[ply].fullmove;

  u64 from = 1ULL << m.from();
  u64 to = 1ULL << m.to();
  Pc p = P;
  if (m.prom()) {
    bb[stm][m.prom()] &= ~to;
    bb[stm][P] |= from;
  } else {
    for (int i = P; i <= K; ++i)
      if (bb[stm][i] & to) { p = (Pc)i; break; }
    bb[stm][p] &= ~to;
    bb[stm][p] |= from;
  }

  if (hist[ply].capSq != 255) bb[1 - stm][hist[ply].capPc] |= 1ULL << hist[ply].capSq;

  if (p == K && !m.prom()) {
    if (m.from() == E1 && m.to() == G1) { bb[WHITE][R] &= ~(1ULL << F1); bb[WHITE][R] |= 1ULL << H1; }
    else if (m.from() == E1 && m.to() == C1) { bb[WHITE][R] &= ~(1ULL << D1); bb[WHITE][R] |= 1ULL << A1; }
    else if (m.from() == E8 && m.to() == G8) { bb[BLACK][R] &= ~(1ULL << F8); bb[BLACK][R] |= 1ULL << H8; }
    else if (m.from() == E8 && m.to() == C8) { bb[BLACK][R] &= ~(1ULL << D8); bb[BLACK][R] |= 1ULL << A8; }
  }

  for (int c = WHITE; c <= BLACK; ++c) {
    occ[c] = 0;
    for (int i = P; i <= K; ++i) occ[c] |= bb[c][i];
  }

  zh = zho;
}

bool onBoard(int f, int r) { return f >= 0 && f < 8 && r >= 0 && r < 8; }
bool occupied(const Board& b, Sq sq) { return ((b.occ[WHITE] | b.occ[BLACK]) >> sq) & 1; }
bool ownPiece(const Board& b, Co c, Sq sq) { return (b.occ[c] >> sq) & 1; }
bool enemyPiece(const Board& b, Co c, Sq sq) { return (b.occ[1 - c] >> sq) & 1; }

int pieceAt(const Board& b, Co c, Sq sq) {
  u64 mask = 1ULL << sq;
  for (int p = P; p <= K; ++p)
    if (b.bb[c][p] & mask) return p;
  return -1;
}

bool isAttacked(const Board& b, Sq sq, Co by) {
  int file = sq % 8;
  int rank = sq / 8;
  auto has = [&](Pc p, int f, int r) {
    return onBoard(f, r) && (b.bb[by][p] & (1ULL << (r * 8 + f)));
  };

  int pawnRank = by == WHITE ? rank - 1 : rank + 1;
  if (has(P, file - 1, pawnRank) || has(P, file + 1, pawnRank)) return true;
  for (auto& d : KNIGHT_DIRS)
    if (has(N, file + d[0], rank + d[1])) return true;
  for (auto& d : KING_DIRS)
    if (has(K, file + d[0], rank + d[1])) return true;

  for (int i = 0; i < 8; i++) {
    const int* d = QUEEN_DIRS[i];
    Pc slider = i < 4 ? R : B;
    int f = file + d[0], r = rank + d[1];
    while (onBoard(f, r)) {
      Sq nsq = (Sq)(r * 8 + f);
      if (occupied(b, nsq)) {
        if ((b.bb[by][slider] | b.bb[by][Q]) & (1ULL << nsq)) return true;
        break;
      }
      f += d[0];
      r += d[1];
    }
  }
  return false;
}

bool isCheck(const Board& b) {
  u64 king = b.bb[b.stm][K];
  if (!king) return true;
  return isAttacked(b, (Sq)__builtin_ctzll(king), (Co)(1 - b.stm));
}

bool castlePathClear(const Board& b, Co side, bool kingSide) {
  Co them = (Co)(1 - side);
  if (side == WHITE) {
    if (kingSide)
      return !occupied(b, F1) && !occupied(b, G1) && !isAttacked(b, E1, them) &&
             !isAttacked(b, F1, them) && !isAttacked(b, G1, them);
    return !occupied(b, B1) && !occupied(b, C1) && !occupied(b, D1) && !isAttacked(b, E1, them) &&
           !isAttacked(b, D1, them) && !isAttacked(b, C1, them);
  }
  if (kingSide)
    return !occupied(b, F8) && !occupied(b, G8) && !isAttacked(b, E8, them) &&
           !isAttacked(b, F8, them) && !isAttacked(b, G8, them);
  return !occupied(b, B8) && !occupied(b, C8) && !occupied(b, D8) && !isAttacked(b, E8, them) &&
         !isAttacked(b, D8, them) && !isAttacked(b, C8, them);
}

void genPseudo(const Board& b, MoveList& moves, bool capturesOnly) {
  moves.clear();
  Co side = b.stm;
  u64 all = b.occ[WHITE] | b.occ[BLACK];

  for (int pi = P; pi <= K; ++pi) {
    Pc p = (Pc)pi;
    u64 pieces = b.bb[side][p];
    while (pieces) {
      Sq from = (Sq)__builtin_ctzll(pieces);
      pieces &= pieces - 1;
      int f = from % 8;
      int r = from / 8;

      if (p == P) {
        int dir = side == WHITE ? 1 : -1;
        int startRank = side == WHITE ? 1 : 6;
        int promoteRank = side == WHITE ? 7 : 0;
        int nr = r + dir;

        if (onBoard(f, nr) && !(all & (1ULL << (nr * 8 + f)))) {
          Sq to = (Sq)(nr * 8 + f);
          if (nr == promoteRank) {
            for (Pc pr : {Q, R, B, N}) moves.push(Move(from, to, pr));
          } else if (!capturesOnly) {
            moves.push(Move(from, to));
            int nr2 = r + 2 * dir;
            if (r == startRank && !(all & (1ULL << (nr2 * 8 + f)))) moves.push(Move(from, (Sq)(nr2 * 8 + f)));
          }
        }

        for (int df : {-1, 1}) {
          int nf = f + df;
          int cr = r + dir;
          if (!onBoard(nf, cr)) continue;
          Sq to = (Sq)(cr * 8 + nf);
          if (!enemyPiece(b, side, to)) continue;
          if (cr == promoteRank) {
            for (Pc pr : {Q, R, B, N}) moves.push(Move(from, to, pr));
          } else {
            moves.push(Move(from, to));
          }
        }

        if (b.ep != A1 && ((side == WHITE && r == 4) || (side == BLACK && r == 3))) {
          int epf = b.ep % 8;
          int epr = b.ep / 8;
          if (abs(epf - f) == 1 && epr == (side == WHITE ? 5 : 2)) moves.push(Move(from, b.ep));
        }
        continue;
      }

      if (p == N || p == K) {
        const int (*dirs)[2] = p == N ? KNIGHT_DIRS : KING_DIRS;
        for (int i = 0; i < 8; i++) {
          int nf = f + dirs[i][0];
          int nr = r + dirs[i][1];
          if (!onBoard(nf, nr)) continue;
          Sq to = (Sq)(nr * 8 + nf);
          if (ownPiece(b, side, to)) continue;
          if (capturesOnly && !enemyPiece(b, side, to)) continue;
          moves.push(Move(from, to));
        }
        if (p == K && !capturesOnly) {
          if (side == WHITE && from == E1) {
            if ((b.castle & 1) && (b.bb[WHITE][R] & (1ULL << H1)) && castlePathClear(b, WHITE, true))
              moves.push(Move(E1, G1));
            if ((b.castle & 2) && (b.bb[WHITE][R] & (1ULL << A1)) && castlePathClear(b, WHITE, false))
              moves.push(Move(E1, C1));
          } else if (side == BLACK && from == E8) {
            if ((b.castle & 4) && (b.bb[BLACK][R] & (1ULL << H8)) && castlePathClear(b, BLACK, true))
              moves.push(Move(E8, G8));
            if ((b.castle & 8) && (b.bb[BLACK][R] & (1ULL << A8)) && castlePathClear(b, BLACK, false))
              moves.push(Move(E8, C8));
          }
        }
        continue;
      }

      const int (*dirs)[2] = p == B ? BISHOP_DIRS : (p == R ? ROOK_DIRS : QUEEN_DIRS);
      int count = p == Q ? 8 : 4;
      for (int i = 0; i < count; i++) {
        int nf = f + dirs[i][0];
        int nr = r + dirs[i][1];
        while (onBoard(nf, nr)) {
          Sq to = (Sq)(nr * 8 + nf);
          if (ownPiece(b, side, to)) break;
          if (!capturesOnly || enemyPiece(b, side, to)) moves.push(Move(from, to));
          if (enemyPiece(b, side, to)) break;
          nf += dirs[i][0];
          nr += dirs[i][1];
        }
      }
    }
  }
}

void filterLegal(const Board& b, MoveList& moves) {
  Board test;
  copyPosition(b, test);
  Co side = b.stm;
  int out = 0;
  for (int i = 0; i < moves.count; i++) {
    Move m = moves.data[i];
    u64 zho = test.zh;
    test.doMove(m);
    u64 king = test.bb[side][K];
    bool legal = king && !isAttacked(test, (Sq)__builtin_ctzll(king), test.stm);
    test.undoMove(m, zho);
    if (legal) moves.data[out++] = m;
  }
  moves.count = out;
}

void genMoves(const Board& b, MoveList& moves) {
  genPseudo(b, moves, false);
  filterLegal(b, moves);
}

void genCaptures(const Board& b, MoveList& moves) {
  genPseudo(b, moves, true);
  filterLegal(b, moves);
}

bool isLegal(const Board& b, Move m) {
  MoveList moves;
  genMoves(b, moves);
  for (const Move& legal : moves)
    if (legal.v == m.v) return true;
  return false;
}

int evaluate(const Board& b) {
  int score = 0;
  for (int c = WHITE; c <= BLACK; ++c) {
    for (int p = P; p <= K; ++p) {
      u64 pieces = b.bb[c][p];
      while (pieces) {
        int sq = __builtin_ctzll(pieces);
        pieces &= pieces - 1;
        int rel = c == WHITE ? sq : (7 - sq / 8) * 8 + sq % 8;
        int v = (p == K ? 0 : PIECE_VALUE[p]) + PST[p][rel];
        score += c == WHITE ? v : -v;
      }
    }
  }
  return (b.stm == WHITE ? score : -score) + 10;
}

struct TEntry {
  u64 zh = 0;
  int score = 0;
  Move best;
  u8 depth = 0;
  u8 flag = 0;
  u8 age = 0;
};

struct TCluster {
  TEntry e[4];
};

std::vector<TCluster> tt;
long long searchNodes = 0;
std::chrono::high_resolution_clock::time_point searchStart;
int searchTimeLimit = 0;
static int moveOverheadMs = 20;

void setMoveOverhead(int ms) { moveOverheadMs = ms; }
u8 ttAge = 1;
Move rootBest;
Move killers[128][2];
Move counterMoves[64 * 64];
int historyTable[2 * 64 * 64];
int staticEvalHist[128];

void setHashMb(size_t mb) {
  size_t clusters = mb * 1024ULL * 1024ULL / sizeof(TCluster);
  if (clusters < 256) clusters = 256;
  tt.assign(clusters, TCluster{});
}

void clearSearch() {
  std::fill(tt.begin(), tt.end(), TCluster{});
  memset(historyTable, 0, sizeof(historyTable));
  memset(killers, 0, sizeof(killers));
  memset(counterMoves, 0, sizeof(counterMoves));
  memset(staticEvalHist, 0, sizeof(staticEvalHist));
}

bool timeUp() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now - searchStart).count() >= searchTimeLimit;
}

TEntry* ttProbe(u64 zh) {
  for (auto& e : tt[zh % tt.size()].e)
    if (e.zh == zh) return &e;
  return nullptr;
}

void ttStore(u64 zh, int score, u8 depth, u8 flag, Move best) {
  TCluster& c = tt[zh % tt.size()];
  TEntry* victim = &c.e[0];
  int victimScore = INF;
  for (auto& e : c.e) {
    if (e.zh == zh || e.zh == 0) { victim = &e; victimScore = -INF; break; }
    int s = e.depth * 4 - (int)(u8)(ttAge - e.age);
    if (s < victimScore) { victim = &e; victimScore = s; }
  }
  if (victim->zh != zh && victim->zh != 0 && depth + 1 < victim->depth) return;
  *victim = {zh, score, best, depth, flag, ttAge};
}

bool isCapture(const Board& b, Move m) {
  if (b.occ[1 - b.stm] & (1ULL << m.to())) return true;
  return b.ep != A1 && m.to() == b.ep && (b.bb[b.stm][P] & (1ULL << m.from()));
}

int capturedValue(const Board& b, Move m) {
  int victim = pieceAt(b, (Co)(1 - b.stm), m.to());
  if (victim < 0 && b.ep != A1 && m.to() == b.ep) victim = P;
  int value = victim >= 0 ? PIECE_VALUE[victim] : 0;
  if (m.prom()) value += PIECE_VALUE[m.prom()] - PIECE_VALUE[P];
  return value;
}

int seeGain(const Board& b, Move m, int depth = 0);

int seeGain(const Board& b, Move m, int depth) {
  int gain = capturedValue(b, m);
  if (gain <= 0 || depth >= 6) return gain;

  Board next;
  copyPosition(b, next);
  next.doMove(m);

  MoveList replies;
  genCaptures(next, replies);
  int bestReply = 0;
  for (const Move& reply : replies) {
    if (reply.to() != m.to()) continue;
    int replyGain = seeGain(next, reply, depth + 1);
    if (replyGain > bestReply) bestReply = replyGain;
  }
  return gain - bestReply;
}

int moveScore(Move m, const Board& b, Move ttMove, int ply, Move prevMove) {
  if (m.v == ttMove.v) return 100000000;
  if (isCapture(b, m)) {
    int victim = pieceAt(b, (Co)(1 - b.stm), m.to());
    if (victim < 0) victim = P;
    int attacker = pieceAt(b, b.stm, m.from());
    return 1000000 + 10 * PIECE_VALUE[victim] - (attacker >= 0 ? PIECE_VALUE[attacker] : 0);
  }
  if (m.prom()) return 900000 + PIECE_VALUE[m.prom()];
  if (ply < 128 && m.v == killers[ply][0].v) return 800000;
  if (ply < 128 && m.v == killers[ply][1].v) return 700000;
  if (prevMove.v && m.v == counterMoves[prevMove.from() * 64 + prevMove.to()].v) return 600000;
  return historyTable[(b.stm * 64 + m.from()) * 64 + m.to()];
}

void orderMoves(MoveList& moves, const Board& b, Move ttMove, int ply, Move prevMove) {
  std::sort(moves.begin(), moves.end(), [&](Move a, Move c) {
    return moveScore(a, b, ttMove, ply, prevMove) > moveScore(c, b, ttMove, ply, prevMove);
  });
}

void updateHistory(Co side, Move m, int bonus) {
  int& h = historyTable[(side * 64 + m.from()) * 64 + m.to()];
  h += bonus - h * abs(bonus) / 16384;
}

bool isDraw(const Board& b) {
  if (b.halfmove >= 100) return true;
  int start = std::max(0, (int)b.ply - (int)b.halfmove);
  for (int i = (int)b.ply - 2; i >= start; i -= 2)
    if (b.hist[i].zh == b.zh) return true;
  return false;
}

int nonPawnMaterial(const Board& b, Co c) {
  return __builtin_popcountll(b.bb[c][N]) * PIECE_VALUE[N] +
         __builtin_popcountll(b.bb[c][B]) * PIECE_VALUE[B] +
         __builtin_popcountll(b.bb[c][R]) * PIECE_VALUE[R] +
         __builtin_popcountll(b.bb[c][Q]) * PIECE_VALUE[Q];
}

int qsearch(Board& b, int alpha, int beta) {
  if ((++searchNodes & 255) == 0 && timeUp()) return evaluate(b);
  if (b.ply >= 1023) return evaluate(b);
  if (isDraw(b)) return 0;

  if (isCheck(b)) {
    MoveList evasions;
    genMoves(b, evasions);
    if (evasions.empty()) return -MATE;
    orderMoves(evasions, b, Move(), 0, Move());
    for (const Move& m : evasions) {
      u64 zho = b.zh;
      b.doMove(m);
      int score = -qsearch(b, -beta, -alpha);
      b.undoMove(m, zho);
      if (score >= beta) return beta;
      if (score > alpha) alpha = score;
    }
    return alpha;
  }

  int standPat = evaluate(b);
  if (standPat >= beta) return beta;
  if (standPat > alpha) alpha = standPat;

  MoveList moves;
  genCaptures(b, moves);
  orderMoves(moves, b, Move(), 0, Move());

  for (const Move& m : moves) {
    if (isCapture(b, m)) {
      int see = seeGain(b, m);
      if (see < 0) continue;
      if (standPat + see + 90 < alpha) continue;
    }
    u64 zho = b.zh;
    b.doMove(m);
    int score = -qsearch(b, -beta, -alpha);
    b.undoMove(m, zho);
    if (score >= beta) return beta;
    if (score > alpha) alpha = score;
  }
  return alpha;
}

int negamax(Board& b, int depth, int alpha, int beta, int ply, Move prevMove) {
  if ((++searchNodes & 255) == 0 && timeUp()) return evaluate(b);
  if (ply >= 127 || b.ply >= 1023) return evaluate(b);
  if (isDraw(b)) return 0;

  bool inCheck = isCheck(b);
  if (depth <= 0 && !inCheck) return qsearch(b, alpha, beta);
  if (inCheck) depth++;

  int staticEval = evaluate(b);
  int alphaOrig = alpha;
  Move ttMove;

  if (TEntry* e = ttProbe(b.zh)) {
    ttMove = e->best;
    if (e->depth >= depth) {
      if (e->flag == 0) return e->score;
      if (e->flag == 1 && e->score >= beta) return e->score;
      if (e->flag == 2 && e->score <= alpha) return e->score;
    }
  }

  if (!inCheck && depth >= 3 && staticEval >= beta && nonPawnMaterial(b, b.stm) > 0 && beta < MATE - 1000) {
    Board nullBoard = b;
    nullBoard.stm = (Co)(1 - nullBoard.stm);
    nullBoard.ep = A1;
    nullBoard.zh = calcHash(nullBoard);
    int reduction = 2 + depth / 4;
    if (-negamax(nullBoard, depth - 1 - reduction, -beta, -beta + 1, ply + 1, Move()) >= beta) return beta;
  }

  if (!inCheck && depth <= 3 && staticEval - 120 * depth >= beta) return beta;

  staticEvalHist[ply] = staticEval;
  bool improving = ply >= 2 && staticEvalHist[ply] >= staticEvalHist[ply - 2];

  MoveList moves;
  genMoves(b, moves);
  if (moves.empty()) return inCheck ? -MATE + ply : 0;
  orderMoves(moves, b, ttMove, ply, prevMove);

  Move bestMove;
  Move quietsSearched[MoveList::MAX];
  int best = -INF;
  int moveCount = 0;
  int quietCount = 0;
  bool searchedMove = false;

  for (const Move& m : moves) {
    Co side = b.stm;
    bool capture = isCapture(b, m);
    if (moveCount > 0 && !inCheck && !capture && !m.prom() && depth <= 2 &&
        staticEval + 90 * depth <= alpha && !improving) {
      moveCount++;
      continue;
    }

    u64 zho = b.zh;
    b.doMove(m);
    searchedMove = true;

    int score;
    if (moveCount == 0) {
      score = -negamax(b, depth - 1, -beta, -alpha, ply + 1, m);
    } else {
      int reduction = 0;
      if (depth >= 3 && moveCount >= 4 && !inCheck && !capture && !m.prom())
        reduction = 1 + (!improving && depth >= 5 && moveCount >= 8);
      score = -negamax(b, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, m);
      if (score > alpha && reduction) score = -negamax(b, depth - 1, -alpha - 1, -alpha, ply + 1, m);
      if (score > alpha && score < beta) score = -negamax(b, depth - 1, -beta, -alpha, ply + 1, m);
    }

    b.undoMove(m, zho);
    moveCount++;
    if (!capture && !m.prom() && quietCount < MoveList::MAX) quietsSearched[quietCount++] = m;

    if (score > best) {
      best = score;
      bestMove = m;
    }
    if (score > alpha) alpha = score;

    if (alpha >= beta) {
      if (!capture) {
        if (ply < 128 && killers[ply][0].v != m.v) {
          killers[ply][1] = killers[ply][0];
          killers[ply][0] = m;
        }
        int bonus = depth * depth * 32;
        updateHistory(side, m, bonus);
        if (prevMove.v) counterMoves[prevMove.from() * 64 + prevMove.to()] = m;
        for (int i = 0; i < quietCount; i++)
          if (quietsSearched[i].v != m.v) updateHistory(side, quietsSearched[i], -bonus);
      }
      ttStore(b.zh, beta, depth, 1, m);
      return beta;
    }
  }

  if (!searchedMove) return staticEval;
  ttStore(b.zh, best, depth, best <= alphaOrig ? 2 : 0, bestMove);
  return best;
}

Move bestMove(Board& b, int timeMs, int maxDepth) {
  if (tt.empty()) setHashMb(32);
  searchStart = std::chrono::high_resolution_clock::now();
  searchTimeLimit = std::max(10, timeMs - moveOverheadMs);
  searchNodes = 0;
  rootBest = Move();
  if (++ttAge == 0) ttAge = 1;
  memset(staticEvalHist, 0, sizeof(staticEvalHist));

  MoveList moves;
  genMoves(b, moves);
  if (moves.empty()) return Move();

  std::vector<int> rootScores(moves.size(), 0);
  std::vector<int> order(moves.size());
  rootBest = moves[0];
  int lastScore = 0;
  maxDepth = std::max(1, std::min(maxDepth, 64));

  for (int depth = 1; depth <= maxDepth; depth++) {
    int window = 35;
    while (true) {
      if (timeUp()) return rootBest;
      int alpha = depth >= 4 ? lastScore - window : -INF;
      int beta = depth >= 4 ? lastScore + window : INF;
      int alphaStart = alpha;
      int betaStart = beta;
      Move bestThisDepth = rootBest;
      int bestScore = -INF;

      for (int i = 0; i < moves.size(); i++) order[i] = i;
      std::stable_sort(order.begin(), order.end(), [&](int a, int c) {
        return rootScores[a] > rootScores[c];
      });

      for (int oi = 0; oi < (int)order.size(); oi++) {
        Move m = moves[order[oi]];
        u64 zho = b.zh;
        b.doMove(m);
        int score;
        if (oi == 0) {
          score = -negamax(b, depth - 1, -beta, -alpha, 1, m);
        } else {
          score = -negamax(b, depth - 1, -alpha - 1, -alpha, 1, m);
          if (score > alpha && score < beta) score = -negamax(b, depth - 1, -beta, -alpha, 1, m);
        }
        b.undoMove(m, zho);

        if (timeUp()) return rootBest;
        rootScores[order[oi]] = score;
        if (score > bestScore) {
          bestScore = score;
          bestThisDepth = m;
        }
        if (score > alpha) alpha = score;
      }

      if (bestScore <= alphaStart || bestScore >= betaStart) {
        window *= 2;
        if (timeUp()) return rootBest;
        continue;
      }

      rootBest = bestThisDepth;
      lastScore = bestScore;
      break;
    }
    if (timeUp()) break;
  }

  return rootBest;
}

std::string squareName(Sq sq) {
  std::string s;
  s += char('a' + sq % 8);
  s += char('1' + sq / 8);
  return s;
}

std::string moveToUci(Move m) {
  if (m.v == 0) return "0000";
  std::string s = squareName(m.from()) + squareName(m.to());
  if (m.prom()) s += "pnbrqk"[m.prom()];
  return s;
}

Move uciToMove(const Board& b, const std::string& uci) {
  if (uci.size() < 4 || uci == "0000") return Move();
  if (uci[0] < 'a' || uci[0] > 'h' || uci[2] < 'a' || uci[2] > 'h') return Move();
  if (uci[1] < '1' || uci[1] > '8' || uci[3] < '1' || uci[3] > '8') return Move();
  Sq from = (Sq)((uci[0] - 'a') + (uci[1] - '1') * 8);
  Sq to = (Sq)((uci[2] - 'a') + (uci[3] - '1') * 8);
  Pc prom = P;
  if (uci.size() >= 5) {
    char c = tolower(uci[4]);
    prom = c == 'n' ? N : c == 'b' ? B : c == 'r' ? R : c == 'q' ? Q : P;
  }
  MoveList moves;
  genMoves(b, moves);
  for (const Move& m : moves)
    if (m.from() == from && m.to() == to && (m.prom() == prom || (!m.prom() && prom == P))) return m;
  return Move();
}

std::string trim(std::string s) {
  while (!s.empty() && isspace((unsigned char)s.front())) s.erase(s.begin());
  while (!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
  return s;
}

std::string lower(std::string s) {
  for (char& c : s) c = tolower((unsigned char)c);
  return s;
}
