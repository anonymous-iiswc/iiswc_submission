# Makefile: BitSIMD-V basic tests

include ../Makefile.common

EXEC := bitsimd-v-basic.out
SRC := bitsimd-v-basic.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

