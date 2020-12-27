CC=gcc


TARGET=alexnet
OBJDIR=./obj/
SRC=./src/
CLFAGS=-w -lm -lpthread 

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


clean: 
	rm -rf $(OBJDIR) $(TARGET)
