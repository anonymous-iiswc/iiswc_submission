# Makefile: C++ version of vector add

include ../Makefile.common

EXEC := gemm.out
SRC := gemm.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

