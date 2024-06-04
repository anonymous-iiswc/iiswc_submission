# Makefile: Radix Sort

include ../Makefile.common

EXEC := radix-sort.out
SRC := radix-sort.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS)  -o $@

clean:
	rm -rf $(EXEC) *.dSYM

