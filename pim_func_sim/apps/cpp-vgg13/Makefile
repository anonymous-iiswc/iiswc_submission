# Makefile: C++ version of vgg

include ../Makefile.common

EXEC := vgg13.out
SRC := vgg.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -fopenmp -o $@

clean:
	rm -rf $(EXEC) *.dSYM

