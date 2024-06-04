# Makefile: C++ version of convolution

include ../Makefile.common

EXEC := conv.out
SRC := conv.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM

