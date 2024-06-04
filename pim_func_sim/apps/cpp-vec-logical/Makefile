# Makefile: C++ version of vector arithmetic

include ../Makefile.common

EXEC := vec-logical.out
SRC := vec-logical.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

