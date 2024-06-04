# Makefile: C++ version of vector addition

include ../Makefile.common

EXEC := vec-xor.out
SRC := vec-xor.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

