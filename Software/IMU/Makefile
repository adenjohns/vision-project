CC = g++
CXXFLAGS = -Wall -std=c++11
INCLUDES = -I.
LIBS = -lpigpio -lrt -pthread
LIBPATH = -L.

# Main target
all: imuControl

# Build the IMU controller program
imuControl: imuControl.cpp RPI_BNO055.cpp
	$(CC) $(CXXFLAGS) $(INCLUDES) -o imuControl imuControl.cpp RPI_BNO055.cpp $(LIBPATH) $(LIBS)

# Clean up build artifacts
clean:
	rm -f imuControl

# Install (optional)
install: imuControl
	cp imuControl /usr/local/bin/

.PHONY: all clean install 
