# Makefile: C++ version of vector boradcast and popcount

include ../Makefile.common

EXEC := popcount.out
SRC := popcount.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

