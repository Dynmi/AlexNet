CC=gcc


TARGET=alexnet
OBJDIR=./obj/
SRC=./src/

# AUGMENT: random horizontal flip on training batches (1/0). See src/train.c.
AUGMENT ?= 1
# OUT_LAYER: number of output classes (10 for CIFAR-10, 100 for CIFAR-100). See src/alexnet.h.
OUT_LAYER ?= 10

CLFAGS=-O2 -w -lm -lpthread -g -DAUGMENT=$(AUGMENT) -DOUT_LAYER=$(OUT_LAYER)

LAYER_OBJ=activation_layer.o batchnorm_layer.o convolution_layer.o dropout_layer.o maxpooling_layer.o fc_layer.o
LAYER_OBJS = $(addprefix $(OBJDIR), $(LAYER_OBJ))

TAR_OBJ=train.o inference.o data.o
TAR_OBJS = $(addprefix $(OBJDIR), $(TAR_OBJ))


$(OBJDIR):
	mkdir -p $(OBJDIR) 

$(TARGET): $(TAR_OBJS) $(LAYER_OBJS) 
	$(CC) -o $(OBJDIR)matrix.o -c $(SRC)matrix.c $(CLFAGS)
	$(CC) -o $@	$(SRC)alexnet.c $(TAR_OBJS) $(LAYER_OBJS) $(OBJDIR)matrix.o $(CLFAGS)

$(TAR_OBJS):
	$(CC) -o $(OBJDIR)train.o -c $(SRC)train.c $(CLFAGS)
	$(CC) -o $(OBJDIR)inference.o -c $(SRC)inference.c $(CLFAGS)
	$(CC) -o $(OBJDIR)data.o -c $(SRC)data.c $(CLFAGS)

$(LAYER_OBJS):
	$(CC) -o $(OBJDIR)activation_layer.o -c $(SRC)activation_layer.c $(CLFAGS)
	$(CC) -o $(OBJDIR)batchnorm_layer.o -c $(SRC)batchnorm_layer.c $(CLFAGS)
	$(CC) -o $(OBJDIR)convolution_layer.o -c $(SRC)convolution_layer.c $(CLFAGS)
	$(CC) -o $(OBJDIR)dropout_layer.o -c $(SRC)dropout_layer.c $(CLFAGS)
	$(CC) -o $(OBJDIR)maxpooling_layer.o -c $(SRC)maxpooling_layer.c $(CLFAGS)
	$(CC) -o $(OBJDIR)fc_layer.o -c $(SRC)fc_layer.c $(CLFAGS)


all: $(OBJDIR) $(TARGET)

single_conv: $(LAYER_OBJS) $(OBJDIR)data.o $(OBJDIR)matrix.o
	$(CC) -O2 -o single_conv $(SRC)single_conv.c \
		$(OBJDIR)data.o $(OBJDIR)convolution_layer.o $(OBJDIR)matrix.o \
		-I$(SRC) $(CLFAGS)

single_fc: $(LAYER_OBJS) $(OBJDIR)data.o $(OBJDIR)matrix.o
	$(CC) -O2 -o single_fc $(SRC)single_fc.c \
		$(OBJDIR)data.o $(OBJDIR)fc_layer.o $(OBJDIR)matrix.o \
		-I$(SRC) $(CLFAGS)

$(OBJDIR)matrix.o:
	$(CC) -o $(OBJDIR)matrix.o -c $(SRC)matrix.c $(CLFAGS)

$(OBJDIR)data.o:
	$(CC) -o $(OBJDIR)data.o -c $(SRC)data.c $(CLFAGS)


clean: 
	rm -rf $(OBJDIR) $(TARGET) single_conv single_fc
