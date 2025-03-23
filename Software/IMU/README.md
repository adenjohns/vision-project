# Homemade BNO055 Sensor library for Raspberry Pi using pigpio


Before using the IMU for anything, you have to edit the Raspberry Pi config file found
at /boot/firmware/config.txt and add this line: dtoverlay=i2c-gpio,bus=1,i2c_gpio_sda=02,i2c_gpio_scl=03
to the bottom (you only have to do this once). This is how we get around the clock stretching problem
that the Raspberry Pi has with I2C

IF CODE DOESN'T WORK:

1. Make sure I2C is enabled on the Raspberry Pi:
   ```
   sudo raspi-config
   ```
   Navigate to "Interface Options" > "I2C" and enable it.

2. Install required packages:
   ```
   sudo apt-get update
   sudo apt-get install -y i2c-tools
   ```

3. Install pigpio library:
   ```
   sudo apt-get install -y pigpio python-pigpio python3-pigpio
   ```

4. Start the pigpio daemon (mando):
   ```
   sudo pigpiod
   ```
   
   To automatically start the pigpio daemon at boot:
   ```
   sudo systemctl enable pigpiod
   ```

5. Verify the IMU is detected by the Pi:
   ```
   i2cdetect -y -r 1
   ```
   The IMU will be at address 0x28

## Building

Install the build tools (probably already have them but just make sure):

```
sudo apt-get install -y build-essential 
```

Then, simply run `make` to build the example program:

```
make
```

## Run the code

pigpio daemon must be running:

```
sudo pigpiod

```
then run the actual code:

```
sudo ./imuControl

```


The code should display:
- Sensor information
- Current calibration status (system, gyroscope, accelerometer, magnetometer)
- Orientation data (Euler angles)
- Temperature

## Calibration

to Calibrate the BNO055:

1. Gyroscope: Place the sensor on a stable surface for a few seconds
2. Accelerometer: Move the sensor into 6 different stable positions (all sides of a cube)
3. Magnetometer: Move the sensor in a figure-8 pattern until calibrated

 A value of 3 means fully calibrated

## All library Functions

The functions this library provides:

- `begin()` - Initialize the sensor
- `getVector()` - Get vector data (accelerometer, gyroscope, etc.)
- `getQuat()` - Get quaternion data
- `getTemp()` - Get temperature
- `getCalibration()` - Get calibration status
- `isFullyCalibrated()` - Check if sensor is fully calibrated
- `setExtCrystalUse()` - Use external crystal for better accuracy
