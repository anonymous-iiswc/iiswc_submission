# Makefile: C++ version of vector add

include ../Makefile.common

EXEC := gemv.out
SRC := gemv.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

