# Makefile: C++ version of filter-by-key

include ../Makefile.common

EXEC := db.out
SRC := db.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

