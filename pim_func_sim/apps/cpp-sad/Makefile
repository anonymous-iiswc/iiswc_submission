# Makefile: C++ version of SAD

include ../Makefile.common

EXEC := sad.out
SRC := sad.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

