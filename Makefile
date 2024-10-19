GCC = g++
MPI = mpigxx

riemann: riemann.cpp
	$(MPI) $(CFLAGS) -o $@ riemann.cpp

hello_riemann: hello_riemann.cpp
	$(GCC) $(CFLAGS) -o $@ hello_riemann.cpp -lm

