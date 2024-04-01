build:
	nvcc -I /usr/include/opencv4 -L /usr/lib/x86_64-linux-gnu/ -L /usr/lib/x86_64-linux-gnu/ -lcurl -ltiff rgb2gray.cu -o rgb2gray -lopencv_core -lopencv_imgcodecs -lopencv_highgui

run:
	./rgb2gray puppy.jpg