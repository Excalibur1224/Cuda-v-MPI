CC = nvcc

cintegrate: cintegrate.cu
	$(CC) $(CFLAGS) -o $@ cintegrate.cu -lm
