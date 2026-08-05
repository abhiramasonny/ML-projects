#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

using u64 = uint64_t;
using u32 = uint32_t;
using u16 = uint16_t;
using u8 = uint8_t;

enum Sq : u8 { A1,B1,C1,D1,E1,F1,G1,H1, A2,B2,C2,D2,E2,F2,G2,H2, A3,B3,C3,D3,E3,F3,G3,H3,
               A4,B4,C4,D4,E4,F4,G4,H4, A5,B5,C5,D5,E5,F5,G5,H5, A6,B6,C6,D6,E6,F6,G6,H6,
               A7,B7,C7,D7,E7,F7,G7,H7, A8,B8,C8,D8,E8,F8,G8,H8 };
enum Pc { P, N, B, R, Q, K };
enum Co { WHITE = 0, BLACK = 1 };

inline constexpr const char* START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

struct Move {
  u16 v = 0;
  Move() = default;
  Move(Sq f, Sq t, Pc p = (Pc)0) : v((f) | ((t) << 6) | ((p) << 12)) {}
  Sq from() const { return (Sq)(v & 63); }
  Sq to() const { return (Sq)((v >> 6) & 63); }
  Pc prom() const { return (Pc)((v >> 12) & 7); }
};

struct MoveList {
  static const int MAX = 256;
  Move data[MAX];
  int count = 0;
  void clear() { count = 0; }
  void push(Move m) { if (count < MAX) data[count++] = m; }
  bool empty() const { return count == 0; }
  int size() const { return count; }
  Move* begin() { return data; }
  Move* end() { return data + count; }
  const Move* begin() const { return data; }
  const Move* end() const { return data + count; }
  Move& operator[](int i) { return data[i]; }
  const Move& operator[](int i) const { return data[i]; }
};

struct Undo {
  u64 zh;
  u32 halfmove;
  u32 fullmove;
  u8 castle;
  Sq ep;
  u8 capSq;
  u8 capPc;
};

struct Board {
  u64 bb[2][6];
  u64 occ[2];
  Co stm;
  u32 ply;
  u64 zh;
  u8 castle;
  Sq ep;
  u32 halfmove;
  u32 fullmove;
  Undo hist[1024];

  void fromFen(const std::string& fen);
  std::string toFen() const;
  void doMove(Move m);
  void undoMove(Move m, u64 zho);
};

void initZobrist();
u64 calcHash(const Board& b);
void copyPosition(const Board& src, Board& dst);

bool onBoard(int f, int r);
bool occupied(const Board& b, Sq sq);
bool ownPiece(const Board& b, Co c, Sq sq);
bool enemyPiece(const Board& b, Co c, Sq sq);
int pieceAt(const Board& b, Co c, Sq sq);

bool isAttacked(const Board& b, Sq sq, Co by);
bool isCheck(const Board& b);
void genMoves(const Board& b, MoveList& moves);
void genCaptures(const Board& b, MoveList& moves);
bool isLegal(const Board& b, Move m);

int evaluate(const Board& b);

void setHashMb(size_t mb);
void setMoveOverhead(int ms);
void clearSearch();
Move bestMove(Board& b, int timeMs, int maxDepth = 64);

std::string squareName(Sq sq);
std::string moveToUci(Move m);
Move uciToMove(const Board& b, const std::string& uci);
std::string trim(std::string s);
std::string lower(std::string s);
