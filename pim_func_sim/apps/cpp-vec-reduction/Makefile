# Makefile: C++ version of vector reduction

include ../Makefile.common

EXEC := vec-red.out
SRC := vec-red.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

