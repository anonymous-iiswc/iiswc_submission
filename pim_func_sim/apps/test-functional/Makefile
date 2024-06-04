# Makefile: Test functional behavior

include ../Makefile.common

EXEC := test-functional.out
SRC := test-functional.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

