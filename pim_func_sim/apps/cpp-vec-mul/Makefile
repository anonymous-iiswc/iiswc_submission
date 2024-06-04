# Makefile: C++ version of vector addition

include ../Makefile.common

EXEC := vec-mul.out
SRC := vec-mul.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

