#include "chess.h"

#include <SDL.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/select.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

u64 perftNodes(Board& b, int depth) {
  if (depth == 0) return 1;
  MoveList moves;
  genMoves(b, moves);
  if (depth == 1) return moves.size();
  u64 total = 0;
  for (const Move& m : moves) {
    u64 zho = b.zh;
    b.doMove(m);
    total += perftNodes(b, depth - 1);
    b.undoMove(m, zho);
  }
  return total;
}

bool sameState(const Board& a, const Board& b) {
  return memcmp(a.bb, b.bb, sizeof(a.bb)) == 0 && memcmp(a.occ, b.occ, sizeof(a.occ)) == 0 &&
         a.stm == b.stm && a.ply == b.ply && a.zh == b.zh && a.castle == b.castle &&
         a.ep == b.ep && a.halfmove == b.halfmove && a.fullmove == b.fullmove;
}

bool verifyMakeUnmake(Board& b, int depth, const std::string& path) {
  if (b.zh != calcHash(b)) {
    std::cout << "hash mismatch before " << path << "\n";
    return false;
  }
  if (depth == 0) return true;

  MoveList moves;
  genMoves(b, moves);
  for (const Move& m : moves) {
    Board before;
    copyPosition(b, before);
    u64 zho = b.zh;
    b.doMove(m);
    std::string next = path.empty() ? moveToUci(m) : path + " " + moveToUci(m);
    if (b.zh != calcHash(b)) {
      std::cout << "hash mismatch after " << next << "\n";
      return false;
    }
    if (!verifyMakeUnmake(b, depth - 1, next)) return false;
    b.undoMove(m, zho);
    if (!sameState(before, b)) {
      std::cout << "make/unmake drift after " << next << "\n";
      return false;
    }
  }
  return true;
}

struct PerftCase {
  const char* name;
  const char* fen;
  int quickDepth;
  std::vector<std::pair<int, u64>> expected;
};

const std::vector<PerftCase> PERFT_CASES = {
  {"startpos", START_FEN, 4, {{1, 20}, {2, 400}, {3, 8902}, {4, 197281}, {5, 4865609}}},
  {"kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/2pPN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3,
   {{1, 48}, {2, 1991}, {3, 95321}, {4, 3856621}}},
  {"en-passant-pins", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4,
   {{1, 14}, {2, 191}, {3, 2812}, {4, 43238}, {5, 674624}}},
  {"promotions", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3,
   {{1, 6}, {2, 264}, {3, 9467}, {4, 422333}}},
  {"middlegame", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3,
   {{1, 46}, {2, 2079}, {3, 89890}, {4, 3894594}}}
};

bool runPerftSuite(bool deep) {
  bool ok = true;
  for (const PerftCase& test : PERFT_CASES) {
    Board b;
    b.fromFen(test.fen);
    if (!verifyMakeUnmake(b, deep ? 3 : 2, test.name)) {
      ok = false;
      continue;
    }
    for (auto& [depth, expected] : test.expected) {
      if (!deep && depth > test.quickDepth) continue;
      Board p;
      p.fromFen(test.fen);
      u64 got = perftNodes(p, depth);
      std::cout << "perft " << test.name << " depth " << depth << " nodes " << got << "\n";
      if (got != expected) {
        std::cout << "expected " << expected << "\n";
        ok = false;
        break;
      }
    }
  }
  std::cout << (ok ? "perft-suite ok\n" : "perft-suite failed\n");
  return ok;
}

std::string joinFen(int argc, char** argv, int start) {
  if (start >= argc) return START_FEN;
  std::string fen;
  for (int i = start; i < argc; i++) {
    if (i > start) fen += ' ';
    fen += argv[i];
  }
  return fen;
}

void runUci() {
  Board b;
  b.fromFen(START_FEN);
  setHashMb(32);

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;

    if (line == "uci") {
      std::cout << "id name Chess Engine\n";
      std::cout << "id author\n";
      std::cout << "option name Hash type spin default 32 min 1 max 4096\n";
      std::cout << "option name Threads type spin default 1 min 1 max 1\n";
      std::cout << "option name Move Overhead type spin default 20 min 0 max 5000\n";
      std::cout << "option name Clear Hash type button\n";
      std::cout << "uciok\n" << std::flush;
    } else if (line == "isready") {
      std::cout << "readyok\n" << std::flush;
    } else if (line == "ucinewgame") {
      b.fromFen(START_FEN);
      clearSearch();
    } else if (line.rfind("setoption", 0) == 0) {
      size_t namePos = line.find(" name ");
      size_t valuePos = line.find(" value ");
      if (namePos == std::string::npos) continue;
      size_t nameEnd = valuePos == std::string::npos ? std::string::npos : valuePos - namePos - 6;
      std::string key = lower(trim(line.substr(namePos + 6, nameEnd)));
      std::string value = valuePos == std::string::npos ? "" : trim(line.substr(valuePos + 7));
      int n = atoi(value.c_str());
      if (key == "hash" && n >= 1 && n <= 4096) setHashMb(n);
      else if (key == "move overhead" && n >= 0 && n <= 5000) setMoveOverhead(n);
      else if (key == "clear hash") clearSearch();
      else if (key == "threads" && n != 1) {
        std::cerr << "fatal: Threads=" << n << " requested, only 1 is supported\n";
        exit(2);
      }
    } else if (line.rfind("position", 0) == 0) {
      if (line.find("startpos") != std::string::npos) {
        b.fromFen(START_FEN);
      } else if (line.find("fen") != std::string::npos) {
        size_t pos = line.find("fen") + 4;
        size_t end = line.find("moves");
        b.fromFen(line.substr(pos, (end == std::string::npos ? line.size() : end) - pos));
      }
      size_t mpos = line.find("moves");
      if (mpos != std::string::npos) {
        std::istringstream iss(line.substr(mpos + 6));
        std::string text;
        while (iss >> text) {
          Move m = uciToMove(b, text);
          if (!m.v) {
            std::cerr << "fatal: invalid move in position command: " << text << "\n";
            exit(2);
          }
          b.doMove(m);
        }
      }
    } else if (line.rfind("go", 0) == 0) {
      int timeMs = 1000;
      int maxDepth = 64;
      std::istringstream iss(line);
      std::string token;
      while (iss >> token) {
        if (token == "movetime") { iss >> timeMs; break; }
        if (token == "depth") {
          int depth;
          iss >> depth;
          maxDepth = std::max(1, std::min(depth, 64));
          timeMs = 24 * 60 * 60 * 1000;
          break;
        }
        if ((b.stm == WHITE && token == "wtime") || (b.stm == BLACK && token == "btime")) {
          int t;
          iss >> t;
          timeMs = std::min(t / 50, 10000);
          break;
        }
      }
      std::cout << "bestmove " << moveToUci(bestMove(b, timeMs, maxDepth)) << "\n" << std::flush;
    } else if (line == "quit") {
      break;
    }
  }
}

std::string moveToSan(const Board& b, Move m) {
  int moved = pieceAt(b, b.stm, m.from());
  if (moved < 0) return moveToUci(m);
  if (moved == K && (m.from() == E1 || m.from() == E8)) {
    if (m.to() == G1 || m.to() == G8) return "O-O";
    if (m.to() == C1 || m.to() == C8) return "O-O-O";
  }
  bool capture = pieceAt(b, (Co)(1 - b.stm), m.to()) >= 0;
  std::string san;
  if (moved != P) san += "PNBRQK"[moved];
  if (moved == P && capture) san += char('a' + m.from() % 8);
  if (capture) san += 'x';
  san += squareName(m.to());
  if (m.prom()) {
    san += '=';
    san += "PNBRQK"[m.prom()];
  }
  return san;
}

Move parseSan(const Board& b, const std::string& raw) {
  std::string s = lower(trim(raw));
  s.erase(std::remove_if(s.begin(), s.end(), [](char c) {
    return c == '+' || c == '#' || c == '!' || c == '?';
  }), s.end());

  MoveList moves;
  genMoves(b, moves);

  if (s == "o-o" || s == "0-0" || s == "o-o-o" || s == "0-0-0") {
    Sq from = b.stm == WHITE ? E1 : E8;
    Sq to = (s == "o-o" || s == "0-0") ? (b.stm == WHITE ? G1 : G8) : (b.stm == WHITE ? C1 : C8);
    for (const Move& m : moves)
      if (m.from() == from && m.to() == to) return m;
    return Move();
  }

  for (const Move& m : moves)
    if (lower(moveToSan(b, m)) == s) return m;
  return Move();
}

void printBoard(const Board& b) {
  for (int r = 7; r >= 0; r--) {
    std::cout << r + 1 << ' ';
    for (int f = 0; f < 8; f++) {
      Sq sq = (Sq)(r * 8 + f);
      char ch = '.';
      for (int c = WHITE; c <= BLACK; ++c) {
        int p = pieceAt(b, (Co)c, sq);
        if (p >= 0) ch = c == WHITE ? "PNBRQK"[p] : tolower("PNBRQK"[p]);
      }
      std::cout << ch << ' ';
    }
    std::cout << "\n";
  }
  std::cout << "  a b c d e f g h\n";
  std::cout << (b.stm == WHITE ? "White" : "Black") << " to move\n";
}

void runCli() {
  Board board;
  board.fromFen(START_FEN);
  setHashMb(32);

  std::cout << "CLI chess. Enter moves like e4, Nf3, O-O, or e2e4.\n";
  std::cout << "Commands: board, fen, reset, quit\n";
  printBoard(board);

  std::string line;
  while (true) {
    std::cout << "> ";
    if (!std::getline(std::cin, line)) break;
    std::string s = lower(trim(line));
    if (s.empty()) continue;

    if (s == "quit" || s == "exit") break;
    if (s == "board") { printBoard(board); continue; }
    if (s == "fen") { std::cout << board.toFen() << "\n"; continue; }
    if (s == "reset") { board.fromFen(START_FEN); printBoard(board); continue; }

    Move userMove = uciToMove(board, s);
    if (!userMove.v) userMove = parseSan(board, line);
    if (!userMove.v) {
      std::cout << "Could not parse that move.\n";
      continue;
    }

    std::cout << "You: " << moveToSan(board, userMove) << " (" << moveToUci(userMove) << ")\n";
    board.doMove(userMove);
    printBoard(board);

    Move engineMove = bestMove(board, 750);
    if (!engineMove.v) {
      std::cout << "Engine has no legal move.\n";
      break;
    }
    std::cout << "Engine: " << moveToSan(board, engineMove) << " (" << moveToUci(engineMove) << ")\n";
    board.doMove(engineMove);
    printBoard(board);
  }
}

const int BOARD_PIXELS = 640;
const int SQ_PIXELS = BOARD_PIXELS / 8;

const char* GLYPH_ROWS[6][7] = {
  {"11110","10001","10001","11110","10000","10000","10000"},
  {"10001","11001","10101","10011","10001","10001","10001"},
  {"11110","10001","10001","11110","10001","10001","11110"},
  {"11110","10001","10001","11110","10100","10010","10001"},
  {"01110","10001","10001","10001","10101","10010","01101"},
  {"10001","10010","10100","11000","10100","10010","10001"}
};

void setColor(SDL_Renderer* r, int red, int green, int blue, int alpha = 255) {
  SDL_SetRenderDrawColor(r, red, green, blue, alpha);
}

void fillCircle(SDL_Renderer* renderer, int cx, int cy, int radius) {
  for (int y = -radius; y <= radius; y++) {
    int span = (int)sqrt(radius * radius - y * y);
    SDL_RenderDrawLine(renderer, cx - span, cy + y, cx + span, cy + y);
  }
}

SDL_Rect rectForSquare(Sq sq) {
  return SDL_Rect{(sq % 8) * SQ_PIXELS, (7 - sq / 8) * SQ_PIXELS, SQ_PIXELS, SQ_PIXELS};
}

void renderBoard(SDL_Renderer* renderer, const Board& b, int selected, Move lastMove) {
  for (int rank = 0; rank < 8; rank++) {
    for (int file = 0; file < 8; file++) {
      SDL_Rect rect = rectForSquare((Sq)(rank * 8 + file));
      if ((rank + file) % 2 == 0) setColor(renderer, 89, 121, 93);
      else setColor(renderer, 237, 227, 203);
      SDL_RenderFillRect(renderer, &rect);
    }
  }

  if (lastMove.v) {
    setColor(renderer, 120, 166, 210);
    SDL_Rect a = rectForSquare(lastMove.from());
    SDL_Rect c = rectForSquare(lastMove.to());
    SDL_RenderFillRect(renderer, &a);
    SDL_RenderFillRect(renderer, &c);
  }

  if (selected >= 0) {
    setColor(renderer, 242, 190, 72);
    SDL_Rect rect = rectForSquare((Sq)selected);
    SDL_RenderFillRect(renderer, &rect);

    MoveList moves;
    genMoves(b, moves);
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    setColor(renderer, 28, 31, 30, 115);
    for (const Move& m : moves) {
      if (m.from() != selected) continue;
      SDL_Rect t = rectForSquare(m.to());
      fillCircle(renderer, t.x + SQ_PIXELS / 2, t.y + SQ_PIXELS / 2, 9);
    }
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_NONE);
  }

  for (int c = WHITE; c <= BLACK; ++c) {
    for (int p = P; p <= K; ++p) {
      u64 pieces = b.bb[c][p];
      while (pieces) {
        Sq sq = (Sq)__builtin_ctzll(pieces);
        pieces &= pieces - 1;
        SDL_Rect rect = rectForSquare(sq);
        int cx = rect.x + SQ_PIXELS / 2;
        int cy = rect.y + SQ_PIXELS / 2;
        if (c == WHITE) setColor(renderer, 244, 241, 231);
        else setColor(renderer, 36, 39, 37);
        fillCircle(renderer, cx, cy, 30);
        if (c == WHITE) setColor(renderer, 32, 36, 34);
        else setColor(renderer, 238, 232, 214);
        for (int row = 0; row < 7; row++) {
          const char* bits = GLYPH_ROWS[p][row];
          for (int col = 0; col < 5; col++) {
            if (bits[col] != '1') continue;
            SDL_Rect cell{cx - 18 + col * 7, cy - 25 + row * 7, 6, 6};
            SDL_RenderFillRect(renderer, &cell);
          }
        }
      }
    }
  }

  SDL_RenderPresent(renderer);
}

void updateTitle(SDL_Window* window, const Board& b, const std::string& suffix) {
  MoveList moves;
  genMoves(b, moves);
  std::string title = "Chess Engine GUI - ";
  if (moves.empty()) title += isCheck(b) ? "checkmate" : "stalemate";
  else title += b.stm == WHITE ? "White to move" : "Engine thinking";
  if (!suffix.empty()) title += " - " + suffix;
  SDL_SetWindowTitle(window, title.c_str());
}

int runGui() {
  if (SDL_Init(SDL_INIT_VIDEO) != 0) return 1;

  SDL_Window* window = SDL_CreateWindow("Chess Engine GUI", SDL_WINDOWPOS_CENTERED,
                                        SDL_WINDOWPOS_CENTERED, BOARD_PIXELS, BOARD_PIXELS,
                                        SDL_WINDOW_SHOWN);
  if (!window) {
    SDL_Quit();
    return 1;
  }

  SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
  if (!renderer) {
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 1;
  }

  Board board;
  board.fromFen(START_FEN);
  setHashMb(32);

  bool running = true;
  int selected = -1;
  Move lastMove;
  updateTitle(window, board, "");
  renderBoard(renderer, board, selected, lastMove);

  while (running) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) running = false;

      if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_r) {
        board.fromFen(START_FEN);
        selected = -1;
        lastMove = Move();
        updateTitle(window, board, "");
      }

      if (event.type != SDL_MOUSEBUTTONDOWN || event.button.button != SDL_BUTTON_LEFT) continue;
      if (board.stm != WHITE) continue;

      Sq clicked = (Sq)((7 - event.button.y / SQ_PIXELS) * 8 + event.button.x / SQ_PIXELS);
      if (selected < 0) {
        if (ownPiece(board, WHITE, clicked)) selected = clicked;
        renderBoard(renderer, board, selected, lastMove);
        continue;
      }

      Move userMove;
      MoveList moves;
      genMoves(board, moves);
      for (const Move& m : moves)
        if (m.from() == selected && m.to() == clicked && (!m.prom() || m.prom() == Q)) userMove = m;

      if (!userMove.v) {
        selected = ownPiece(board, WHITE, clicked) ? clicked : -1;
        renderBoard(renderer, board, selected, lastMove);
        continue;
      }

      board.doMove(userMove);
      lastMove = userMove;
      selected = -1;
      updateTitle(window, board, "you played " + moveToUci(userMove));
      renderBoard(renderer, board, selected, lastMove);

      genMoves(board, moves);
      if (!moves.empty()) {
        Move engineMove = bestMove(board, 600);
        if (engineMove.v) {
          board.doMove(engineMove);
          lastMove = engineMove;
          updateTitle(window, board, "engine played " + moveToUci(engineMove));
        }
      }
      renderBoard(renderer, board, selected, lastMove);
    }
    SDL_Delay(8);
  }

  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}

struct UciProc {
  std::string path;
  std::vector<std::string> args;
  pid_t pid = -1;
  int inFd = -1;
  int outFd = -1;

  UciProc(const std::string& p, const std::vector<std::string>& a) : path(p), args(a) { spawn(); }
  ~UciProc() { terminate(); }

  void spawn() {
    if (pid != -1) return;
    int inPipe[2], outPipe[2];
    if (pipe(inPipe) != 0 || pipe(outPipe) != 0) throw std::runtime_error("pipe failed");

    pid = fork();
    if (pid == 0) {
      dup2(inPipe[0], STDIN_FILENO);
      dup2(outPipe[1], STDOUT_FILENO);
      close(inPipe[0]);
      close(inPipe[1]);
      close(outPipe[0]);
      close(outPipe[1]);
      std::vector<char*> argv;
      argv.push_back(const_cast<char*>(path.c_str()));
      for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
      argv.push_back(nullptr);
      execvp(path.c_str(), argv.data());
      _exit(127);
    }

    close(inPipe[0]);
    close(outPipe[1]);
    inFd = inPipe[1];
    outFd = outPipe[0];
  }

  void terminate() {
    if (pid <= 0) return;
    send("quit");
    for (int i = 0; i < 20; i++) {
      int status = 0;
      if (waitpid(pid, &status, WNOHANG) == pid) {
        pid = -1;
        close(inFd);
        close(outFd);
        return;
      }
      usleep(50000);
    }
    kill(pid, SIGKILL);
    int status = 0;
    waitpid(pid, &status, 0);
    pid = -1;
    close(inFd);
    close(outFd);
  }

  void restart() {
    terminate();
    spawn();
    init();
  }

  void send(const std::string& s) const {
    std::string line = s + "\n";
    size_t written = 0;
    while (written < line.size()) {
      ssize_t n = write(inFd, line.data() + written, line.size() - written);
      if (n < 0) {
        if (errno == EINTR) continue;
        return;
      }
      written += n;
    }
  }

  bool readLine(std::string& line, int timeoutMs = 30000) const {
    line.clear();
    char ch = 0;
    while (true) {
      fd_set fds;
      FD_ZERO(&fds);
      FD_SET(outFd, &fds);
      timeval tv{timeoutMs / 1000, (timeoutMs % 1000) * 1000};
      int ready = select(outFd + 1, &fds, nullptr, nullptr, &tv);
      if (ready < 0) {
        if (errno == EINTR) continue;
        return false;
      }
      if (ready == 0) return false;
      ssize_t n = read(outFd, &ch, 1);
      if (n < 0) {
        if (errno == EINTR) continue;
        return false;
      }
      if (n == 0) return false;
      if (ch == '\n') return true;
      if (ch != '\r') line.push_back(ch);
    }
  }

  void waitFor(const std::string& prefix) const {
    std::string line;
    while (true) {
      if (!readLine(line)) throw std::runtime_error("engine stopped waiting for " + prefix);
      if (line.rfind(prefix, 0) == 0) return;
    }
  }

  void init() const {
    send("uci");
    waitFor("uciok");
    ready();
  }

  void ready() const {
    send("isready");
    waitFor("readyok");
  }

  std::string positionCommand(const std::vector<std::string>& moves) const {
    std::string pos = "position startpos";
    if (!moves.empty()) {
      pos += " moves";
      for (auto& m : moves) pos += " " + m;
    }
    return pos;
  }

  std::string askBestMove(const std::vector<std::string>& moves, int movetimeMs) {
    send(positionCommand(moves));
    send("go movetime " + std::to_string(movetimeMs));
    std::string line;
    while (readLine(line)) {
      if (line.rfind("bestmove ", 0) != 0) continue;
      std::string move = line.substr(9);
      size_t space = move.find(' ');
      if (space != std::string::npos) move.resize(space);
      return move;
    }
    restart();
    return "0000";
  }

  int askEvalCp(const std::vector<std::string>& moves, int depth) {
    send(positionCommand(moves));
    send("go depth " + std::to_string(depth));
    int cp = 0;
    std::string line;
    while (readLine(line)) {
      size_t p = line.find("score cp ");
      if (p != std::string::npos) cp = atoi(line.c_str() + p + 9);
      p = line.find("score mate ");
      if (p != std::string::npos) cp = atoi(line.c_str() + p + 11) > 0 ? 100000 : -100000;
      if (line.rfind("bestmove ", 0) == 0) return cp;
    }
    restart();
    return cp;
  }
};

struct StartPosition {
  const char* name;
  std::vector<std::string> opening;
};

const std::vector<StartPosition> BENCH_POSITIONS = {
  {"startpos", {}},
  {"italian", {"e2e4","e7e5","g1f3","b8c6","f1c4","f8c5"}},
  {"ruy-lopez", {"e2e4","e7e5","g1f3","b8c6","f1b5","a7a6"}},
  {"scotch", {"e2e4","e7e5","g1f3","b8c6","d2d4","e5d4"}},
  {"kings-gambit", {"e2e4","e7e5","f2f4","e5f4"}},
  {"sicilian-najdorf", {"e2e4","c7c5","g1f3","d7d6","d2d4","c5d4","f3d4","g8f6","b1c3","a7a6"}},
  {"sicilian-dragon", {"e2e4","c7c5","g1f3","d7d6","d2d4","c5d4","f3d4","g8f6","b1c3","g7g6"}},
  {"french-advance", {"e2e4","e7e6","d2d4","d7d5","e4e5","c7c5"}},
  {"french-exchange", {"e2e4","e7e6","d2d4","d7d5","e4d5","e6d5"}},
  {"caro-advance", {"e2e4","c7c6","d2d4","d7d5","e4e5","c8f5"}},
  {"caro-classical", {"e2e4","c7c6","d2d4","d7d5","b1c3","d5e4"}},
  {"pirc", {"e2e4","d7d6","d2d4","g8f6","b1c3","g7g6"}},
  {"scandinavian", {"e2e4","d7d5","e4d5","d8d5","b1c3","d5a5"}},
  {"queens-gambit-declined", {"d2d4","d7d5","c2c4","e7e6","b1c3","g8f6"}},
  {"slav", {"d2d4","d7d5","c2c4","c7c6","g1f3","g8f6"}},
  {"kings-indian", {"d2d4","g8f6","c2c4","g7g6","b1c3","f8g7","e2e4","d7d6"}},
  {"nimzo-indian", {"d2d4","g8f6","c2c4","e7e6","b1c3","f8b4"}},
  {"queens-indian", {"d2d4","g8f6","c2c4","e7e6","g1f3","b7b6"}},
  {"grunfeld", {"d2d4","g8f6","c2c4","g7g6","b1c3","d7d5"}},
  {"london", {"d2d4","d7d5","c1f4","g8f6","e2e3","e7e6"}},
  {"english", {"c2c4","e7e5","b1c3","g8f6","g2g3","d7d5"}},
  {"reti", {"g1f3","d7d5","g2g3","g8f6","f1g2","g7g6"}},
  {"bird", {"f2f4","d7d5","g1f3","g8f6","e2e3","g7g6"}},
  {"larsen", {"b2b3","d7d5","c1b2","g8f6","e2e3","e7e6"}},
  {"sokolsky", {"b2b4","e7e5","c1b2","f8b4","b2e5","g8f6"}}
};

struct GameTask {
  int id;
  int anchor;
  bool selfWhite;
  const StartPosition* start;
};

struct FinishedGame {
  GameTask task;
  double score;
  std::string reason;
};

int envInt(const char* name, int fallback, int minValue) {
  const char* raw = getenv(name);
  if (!raw) return fallback;
  return std::max(minValue, atoi(raw));
}

double playBenchGame(UciProc& self, UciProc& opponent, UciProc& adjudicator, bool selfWhite,
                     const StartPosition& start, int movetimeMs, int maxPly, int adjudicationDepth,
                     std::string& reason) {
  Board board;
  board.fromFen(START_FEN);
  std::vector<std::string> uciMoves;
  for (const std::string& text : start.opening) {
    Move m = uciToMove(board, text);
    if (!m.v) {
      reason = "invalid-opening";
      return 0.5;
    }
    uciMoves.push_back(moveToUci(m));
    board.doMove(m);
  }

  self.send("ucinewgame");
  opponent.send("ucinewgame");
  self.ready();
  opponent.ready();

  for (int ply = 0; ply < maxPly; ply++) {
    MoveList legal;
    genMoves(board, legal);
    if (legal.empty()) {
      if (!isCheck(board)) {
        reason = "stalemate";
        return 0.5;
      }
      reason = "checkmate";
      bool whiteMated = board.stm == WHITE;
      return (selfWhite ? !whiteMated : whiteMated) ? 1.0 : 0.0;
    }

    bool selfToMove = (board.stm == WHITE) == selfWhite;
    std::string best = selfToMove ? self.askBestMove(uciMoves, movetimeMs)
                                  : opponent.askBestMove(uciMoves, movetimeMs);
    Move m = uciToMove(board, best);
    if (!m.v) {
      reason = "illegal-move:" + best;
      return selfToMove ? 0.0 : 1.0;
    }
    uciMoves.push_back(moveToUci(m));
    board.doMove(m);
  }

  int cp = adjudicator.askEvalCp(uciMoves, adjudicationDepth);
  if (board.stm == BLACK) cp = -cp;
  if (!selfWhite) cp = -cp;
  reason = "adjudicated:" + std::to_string(cp);
  if (cp > 250) return 1.0;
  if (cp < -250) return 0.0;
  return 0.5;
}

int runBench() {
  signal(SIGPIPE, SIG_IGN);

  const char* selfPathEnv = getenv("BENCH_SELF_PATH");
  const char* stockfishEnv = getenv("BENCH_STOCKFISH_PATH");
  std::string selfPath = selfPathEnv ? selfPathEnv : "./engine";
  std::string stockfishPath = stockfishEnv ? stockfishEnv : "/opt/homebrew/bin/stockfish";

  std::vector<int> anchors;
  int minElo = envInt("BENCH_MIN_ELO", 1320, 1);
  int maxElo = envInt("BENCH_MAX_ELO", 2520, minElo);
  int eloStep = envInt("BENCH_ELO_STEP", 200, 1);
  for (int elo = minElo; elo <= maxElo; elo += eloStep) anchors.push_back(elo);

  int positionLimit = std::min(envInt("BENCH_POSITION_LIMIT", (int)BENCH_POSITIONS.size(), 1),
                               (int)BENCH_POSITIONS.size());
  int movetimeMs = envInt("BENCH_MOVETIME_MS", 40, 1);
  int maxPly = envInt("BENCH_MAX_PLY", 120, 2);
  int adjudicationDepth = envInt("BENCH_ADJ_DEPTH", 8, 1);

  int pairsPerPosition = envInt("BENCH_PAIRS_PER_POSITION", 1, 1);
  if (getenv("BENCH_GAMES")) {
    int requested = std::max(1, atoi(getenv("BENCH_GAMES")));
    int perPass = (int)anchors.size() * positionLimit * 2;
    pairsPerPosition = std::max(1, (requested + perPass - 1) / perPass);
  }

  int workers = envInt("BENCH_WORKERS", std::max(1, std::min(4, (int)sysconf(_SC_NPROCESSORS_ONLN) / 3)), 1);

  std::vector<GameTask> tasks;
  for (int anchor : anchors)
    for (int pair = 0; pair < pairsPerPosition; pair++)
      for (int i = 0; i < positionLimit; i++) {
        tasks.push_back({(int)tasks.size(), anchor, true, &BENCH_POSITIONS[i]});
        tasks.push_back({(int)tasks.size(), anchor, false, &BENCH_POSITIONS[i]});
      }

  std::cout << "benchmark_games " << tasks.size() << " workers " << workers
            << " elos " << anchors.size() << " positions " << positionLimit
            << " movetime_ms " << movetimeMs << " max_ply " << maxPly << "\n" << std::flush;

  std::atomic<int> nextTask{0};
  std::atomic<int> completed{0};
  std::mutex outMutex;
  std::mutex resultsMutex;
  std::vector<FinishedGame> finished;

  auto worker = [&](int workerId) {
    UciProc self(selfPath, {"--uci"});
    UciProc stockfish(stockfishPath, {});
    UciProc adjudicator(stockfishPath, {});
    self.init();
    stockfish.init();
    adjudicator.init();

    while (true) {
      int index = nextTask.fetch_add(1);
      if (index >= (int)tasks.size()) break;
      const GameTask& task = tasks[index];

      stockfish.send("setoption name Threads value 1");
      stockfish.send("setoption name Hash value 16");
      stockfish.send("setoption name UCI_LimitStrength value true");
      stockfish.send("setoption name UCI_Elo value " + std::to_string(task.anchor));
      stockfish.ready();
      adjudicator.send("setoption name UCI_LimitStrength value false");
      adjudicator.send("setoption name Threads value 1");
      adjudicator.send("setoption name Hash value 16");
      adjudicator.ready();

      std::string reason;
      double score = playBenchGame(self, stockfish, adjudicator, task.selfWhite, *task.start,
                                   movetimeMs, maxPly, adjudicationDepth, reason);

      {
        std::lock_guard<std::mutex> lock(resultsMutex);
        finished.push_back({task, score, reason});
      }
      int done = completed.fetch_add(1) + 1;
      {
        std::lock_guard<std::mutex> lock(outMutex);
        std::cout << "done " << done << "/" << tasks.size() << " worker " << workerId
                  << " anchor " << task.anchor << " pos " << task.start->name
                  << " color " << (task.selfWhite ? "white" : "black")
                  << " result " << score << " reason " << reason << "\n" << std::flush;
      }
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < workers; i++) threads.emplace_back(worker, i + 1);
  for (auto& t : threads) t.join();

  std::cout << "\nsummary_by_elo\n";
  std::vector<std::pair<int, std::pair<double, int>>> byElo;
  double totalScore = 0;
  int playedGames = 0;

  for (int anchor : anchors) {
    double score = 0;
    int games = 0;
    int wins = 0, draws = 0, losses = 0;
    for (const FinishedGame& g : finished) {
      if (g.task.anchor != anchor) continue;
      score += g.score;
      games++;
      if (g.score == 1.0) wins++;
      else if (g.score == 0.0) losses++;
      else draws++;
    }
    if (!games) continue;
    byElo.push_back({anchor, {score, games}});
    totalScore += score;
    playedGames += games;
    double adjusted = (score + 0.5) / (games + 1.0);
    std::cout << "vs_sf_elo " << anchor << " score " << score << "/" << games
              << " pct " << score / games << " WDL " << wins << "/" << draws << "/" << losses
              << " elo_from_anchor " << anchor + 400.0 * log10(adjusted / (1.0 - adjusted)) << "\n";
  }

  auto logLikelihood = [&](double engineElo) {
    double ll = 0;
    for (auto& [anchor, sg] : byElo) {
      double p = 1.0 / (1.0 + pow(10.0, (anchor - engineElo) / 400.0));
      p = std::clamp(p, 1e-6, 1.0 - 1e-6);
      ll += sg.first * log(p) + (sg.second - sg.first) * log(1.0 - p);
    }
    return ll;
  };

  double bestElo = 1500;
  double bestLl = -1e100;
  for (double elo = 800; elo <= 3200; elo += 1.0) {
    double ll = logLikelihood(elo);
    if (ll > bestLl) {
      bestLl = ll;
      bestElo = elo;
    }
  }

  double lo = 800, hi = 3200;
  for (double elo = bestElo; elo >= 800; elo -= 1.0)
    if (logLikelihood(elo) < bestLl - 1.92) { lo = elo; break; }
  for (double elo = bestElo; elo <= 3200; elo += 1.0)
    if (logLikelihood(elo) < bestLl - 1.92) { hi = elo; break; }

  std::cout << "\ntotal_score " << totalScore << "/" << playedGames << "\n";
  std::cout << "mle_elo " << bestElo << "\n";
  std::cout << "mle_ci95 " << lo << " " << hi << "\n";
  return 0;
}

int main(int argc, char** argv) {
  initZobrist();
  std::string mode = argc > 1 ? argv[1] : "";

  if (mode == "--uci") {
    runUci();
  } else if (mode == "--gui") {
    return runGui();
  } else if (mode == "--bench") {
    return runBench();
  } else if (mode == "--test") {
    return runPerftSuite(argc > 2 && std::string(argv[2]) == "deep") ? 0 : 1;
  } else if (mode == "--perft") {
    if (argc < 3) {
      std::cerr << "usage: ./engine --perft <depth> [fen]\n";
      return 2;
    }
    Board b;
    b.fromFen(joinFen(argc, argv, 3));
    std::cout << perftNodes(b, atoi(argv[2])) << "\n";
  } else if (mode == "--perft-divide") {
    if (argc < 3) {
      std::cerr << "usage: ./engine --perft-divide <depth> [fen]\n";
      return 2;
    }
    Board b;
    b.fromFen(joinFen(argc, argv, 3));
    MoveList moves;
    genMoves(b, moves);
    u64 total = 0;
    for (const Move& m : moves) {
      u64 zho = b.zh;
      b.doMove(m);
      u64 count = perftNodes(b, atoi(argv[2]) - 1);
      b.undoMove(m, zho);
      total += count;
      std::cout << moveToUci(m) << ": " << count << "\n";
    }
    std::cout << "total: " << total << "\n";
  } else if (mode == "--eval") {
    if (argc < 3) {
      std::cerr << "usage: ./engine --eval <fen>\n";
      return 2;
    }
    Board b;
    b.fromFen(joinFen(argc, argv, 2));
    std::cout << evaluate(b) << "\n";
  } else {
    runCli();
  }
  return 0;
}
