CXXFLAGS = -O3 -march=native -std=c++17 -Wall $(shell pkg-config --cflags sdl2)
LDFLAGS = $(shell pkg-config --libs sdl2)

engine: engine.cpp main.cpp chess.h
	clang++ $(CXXFLAGS) -o engine engine.cpp main.cpp $(LDFLAGS)

test: engine
	./engine --test

test-deep: engine
	./engine --test deep

clean:
	rm -f engine

.PHONY: test test-deep clean
