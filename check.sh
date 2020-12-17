gcc -c alexnet.c data.c train.c -w
gcc -o test test.c alexnet.o data.o train.o -w -lm
./test
