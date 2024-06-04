# Makefile: C++ version of vector addition

include ../Makefile.common

EXEC := lr.out
SRC := lr.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

