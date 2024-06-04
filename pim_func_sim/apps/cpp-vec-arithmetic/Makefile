# Makefile: C++ version of vector arithmetic

include ../Makefile.common

EXEC := vec-ar.out
SRC := vec-arithmetic.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

