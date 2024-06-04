# Makefile: BitSIMD-V basic tests

include ../Makefile.common

EXEC := bitsimd-perf.out
SRC := bitsimd-perf.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

