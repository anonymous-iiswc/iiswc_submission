# Makefile: Database filtering

include ../Makefile.common

EXEC := db-filtering.out
SRC := db-filtering.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS)  -o $@

clean:
	rm -rf $(EXEC) *.dSYM

