# Makefile: C++ version of max pool

include ../Makefile.common

EXEC := pool.out
SRC := pool.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

