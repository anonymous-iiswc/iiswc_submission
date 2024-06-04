# Makefile: C++ version of vector addition

include ../Makefile.common

EXEC := dot-prod.out
SRC := dot-prod.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

