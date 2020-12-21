gcc -c alexnet.c data.c train.c -w -lpthread
gcc -o test test.c alexnet.o data.o train.o -w -lm -lpthread
./test
