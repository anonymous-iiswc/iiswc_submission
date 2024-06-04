# Makefile: C++ version of vector arithmetic

include ../Makefile.common

EXEC := vec-comp.out
SRC := vec-comparator.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

