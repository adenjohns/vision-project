#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "RPI_BNO055.h"

// Constructor
RPI_BNO055::RPI_BNO055(int32_t sensorID, uint8_t address, int i2cBus) {
  _sensorID = sensorID;
  _address = address;
  _i2cBus = i2cBus;
  _i2cHandle = -1;
}

// Destructor
RPI_BNO055::~RPI_BNO055() {
  closePigpio();
}

// Initialize pigpio
bool RPI_BNO055::initPigpio() {
  if (_i2cHandle >= 0) {
    // Already initialized
    return true;
  }
  
  // Initialize pigpio with more detailed error checking
  if (gpioInitialise() < 0) {
    fprintf(stderr, "Failed to initialize pigpio\n");
    return false;
  }
  
  // Print debug info
  fprintf(stderr, "Opening I2C bus %d with address 0x%02X\n", _i2cBus, _address);
  
  // Open I2C bus with flags set to 0
  _i2cHandle = i2cOpen(_i2cBus, _address, 0);
  if (_i2cHandle < 0) {
    fprintf(stderr, "Failed to open I2C device: %d\n", _i2cHandle);
    // Try closing and reopening
    gpioTerminate();
    gpioInitialise();
    _i2cHandle = i2cOpen(_i2cBus, _address, 0);
    if (_i2cHandle < 0) {
      fprintf(stderr, "Second attempt failed: %d\n", _i2cHandle);
      gpioTerminate();
      return false;
    }
  }
  
  fprintf(stderr, "Successfully opened I2C device with handle: %d\n", _i2cHandle);
  return true;
}

// Close pigpio
void RPI_BNO055::closePigpio() {
  if (_i2cHandle >= 0) {
    i2cClose(_i2cHandle);
    _i2cHandle = -1;
    gpioTerminate();
  }
}

// Delay function
void RPI_BNO055::delay(int ms) {
  gpioDelay(ms * 1000); // Convert ms to μs
}

/*!
 *  @brief  Sets up the HW
 *  @param  mode
 *          mode values
 *           [OPERATION_MODE_CONFIG,
 *            OPERATION_MODE_ACCONLY,
 *            OPERATION_MODE_MAGONLY,
 *            OPERATION_MODE_GYRONLY,
 *            OPERATION_MODE_ACCMAG,
 *            OPERATION_MODE_ACCGYRO,
 *            OPERATION_MODE_MAGGYRO,
 *            OPERATION_MODE_AMG,
 *            OPERATION_MODE_IMUPLUS,
 *            OPERATION_MODE_COMPASS,
 *            OPERATION_MODE_M4G,
 *            OPERATION_MODE_NDOF_FMC_OFF,
 *            OPERATION_MODE_NDOF]
 *  @return true if successful
 */
bool RPI_BNO055::begin() {
  // Default to NDOF mode
  bno055_opmode_t mode = OPERATION_MODE_NDOF;
  
  // Check if pigpio is initialized
  if (!initPigpio()) {
    return false;
  }
  
  // Wait for BNO055 to boot (can take up to 850ms)
  int timeout = 850;
  bool detected = false;
  while (timeout > 0 && !detected) {
    uint8_t id = read8(BNO055_CHIP_ID_ADDR);
    if (id == BNO055_ID) {
      detected = true;
      break;
    }
    delay(10);
    timeout -= 10;
  }
  
  if (!detected) {
    // Try once more after a delay
    delay(1000);
    uint8_t id = read8(BNO055_CHIP_ID_ADDR);
    if (id != BNO055_ID) {
      return false;  // Sensor not detected
    }
  }
  
  // Switch to config mode
  setMode(OPERATION_MODE_CONFIG);
  
  // Reset the device
  write8(BNO055_SYS_TRIGGER_ADDR, 0x20);
  delay(30);
  
  // Wait for the chip ID to appear again
  timeout = 500;
  while (timeout > 0) {
    uint8_t id = read8(BNO055_CHIP_ID_ADDR);
    if (id == BNO055_ID) {
      break;
    }
    delay(10);
    timeout -= 10;
  }
  
  if (timeout <= 0) {
    return false;  // Reset failed
  }
  
  delay(50);
  
  // Set to normal power mode
  write8(BNO055_PWR_MODE_ADDR, POWER_MODE_NORMAL);
  delay(10);
  
  // Set default page
  write8(BNO055_PAGE_ID_ADDR, 0);
  delay(10);
  
  // Disable system triggers
  write8(BNO055_SYS_TRIGGER_ADDR, 0x0);
  delay(10);
  
  // Set the requested operating mode
  setMode(mode);
  delay(20);
  
  return true;
}

/*!
 *  @brief  Puts the chip in the specified operating mode
 *  @param  mode
 *          mode values
 */
void RPI_BNO055::setMode(bno055_opmode_t mode) {
  _mode = mode;
  write8(BNO055_OPR_MODE_ADDR, mode);
  delay(30);  // Wait for mode change to take effect
}

/*!
 *  @brief  Gets the current operating mode of the chip
 *  @return  operating_mode
 */
bno055_opmode_t RPI_BNO055::getMode() {
  return (bno055_opmode_t)read8(BNO055_OPR_MODE_ADDR);
}

/*!
 *  @brief  Use the external 32.768KHz crystal
 *  @param  usextal
 *          use external crystal boolean
 */
void RPI_BNO055::setExtCrystalUse(bool usextal) {
  bno055_opmode_t modeback = _mode;
  
  // Switch to config mode
  setMode(OPERATION_MODE_CONFIG);
  delay(25);
  
  write8(BNO055_PAGE_ID_ADDR, 0);
  if (usextal) {
    write8(BNO055_SYS_TRIGGER_ADDR, 0x80);
  } else {
    write8(BNO055_SYS_TRIGGER_ADDR, 0x00);
  }
  delay(10);
  
  // Return to previous mode
  setMode(modeback);
  delay(20);
}

/*!
 *  @brief  Gets the temperature in degrees celsius
 *  @return temperature in degrees celsius
 */
int8_t RPI_BNO055::getTemp() {
  return (int8_t)read8(BNO055_TEMP_ADDR);
}

/*!
 *  @brief  Gets the current calibration state
 *  @param  sys
 *          System calibration status
 *  @param  gyro
 *          Gyroscope calibration status
 *  @param  accel
 *          Accelerometer calibration status
 *  @param  mag
 *          Magnetometer calibration status
 */
void RPI_BNO055::getCalibration(uint8_t *sys, uint8_t *gyro, uint8_t *accel, uint8_t *mag) {
  uint8_t calData = read8(BNO055_CALIB_STAT_ADDR);
  
  if (sys != NULL) {
    *sys = (calData >> 6) & 0x03;
  }
  if (gyro != NULL) {
    *gyro = (calData >> 4) & 0x03;
  }
  if (accel != NULL) {
    *accel = (calData >> 2) & 0x03;
  }
  if (mag != NULL) {
    *mag = calData & 0x03;
  }
}

/*!
 *  @brief  Gets a vector reading from the specified source
 *  @param  vector_type
 *          possible vector type values
 *           [VECTOR_ACCELEROMETER
 *            VECTOR_MAGNETOMETER
 *            VECTOR_GYROSCOPE
 *            VECTOR_EULER
 *            VECTOR_LINEARACCEL
 *            VECTOR_GRAVITY]
 *  @return  vector from specified source
 */
imu::Vector<3> RPI_BNO055::getVector(vector_type_t vector_type) {
  imu::Vector<3> xyz;
  uint8_t buffer[6];
  memset(buffer, 0, 6);
  
  // Read vector data (6 bytes)
  readLen((bno055_reg_t)vector_type, buffer, 6);
  
  int16_t x = ((int16_t)buffer[0]) | (((int16_t)buffer[1]) << 8);
  int16_t y = ((int16_t)buffer[2]) | (((int16_t)buffer[3]) << 8);
  int16_t z = ((int16_t)buffer[4]) | (((int16_t)buffer[5]) << 8);
  
  // Convert the value to an appropriate range and assign to vector
  switch (vector_type) {
    case VECTOR_MAGNETOMETER:
      // 1uT = 16 LSB
      xyz[0] = ((double)x) / 16.0;
      xyz[1] = ((double)y) / 16.0;
      xyz[2] = ((double)z) / 16.0;
      break;
    case VECTOR_GYROSCOPE:
      // 1dps = 16 LSB
      xyz[0] = ((double)x) / 16.0;
      xyz[1] = ((double)y) / 16.0;
      xyz[2] = ((double)z) / 16.0;
      break;
    case VECTOR_EULER:
      // 1 degree = 16 LSB
      xyz[0] = ((double)x) / 16.0;
      xyz[1] = ((double)y) / 16.0;
      xyz[2] = ((double)z) / 16.0;
      break;
    case VECTOR_ACCELEROMETER:
      // 1m/s^2 = 100 LSB
      xyz[0] = ((double)x) / 100.0;
      xyz[1] = ((double)y) / 100.0;
      xyz[2] = ((double)z) / 100.0;
      break;
    case VECTOR_LINEARACCEL:
      // 1m/s^2 = 100 LSB
      xyz[0] = ((double)x) / 100.0;
      xyz[1] = ((double)y) / 100.0;
      xyz[2] = ((double)z) / 100.0;
      break;
    case VECTOR_GRAVITY:
      // 1m/s^2 = 100 LSB
      xyz[0] = ((double)x) / 100.0;
      xyz[1] = ((double)y) / 100.0;
      xyz[2] = ((double)z) / 100.0;
      break;
  }
  
  return xyz;
}

/*!
 *  @brief  Gets a quaternion reading from the specified source
 *  @return quaternion reading
 */
imu::Quaternion RPI_BNO055::getQuat() {
  uint8_t buffer[8];
  memset(buffer, 0, 8);
  
  // Read quaternion data (8 bytes)
  readLen(BNO055_QUATERNION_DATA_W_LSB_ADDR, buffer, 8);
  
  int16_t w = (((uint16_t)buffer[1]) << 8) | ((uint16_t)buffer[0]);
  int16_t x = (((uint16_t)buffer[3]) << 8) | ((uint16_t)buffer[2]);
  int16_t y = (((uint16_t)buffer[5]) << 8) | ((uint16_t)buffer[4]);
  int16_t z = (((uint16_t)buffer[7]) << 8) | ((uint16_t)buffer[6]);
  
  // Scale values to proper range
  const double scale = (1.0 / (1 << 14));
  return imu::Quaternion(scale * w, scale * x, scale * y, scale * z);
}

/*!
 *  @brief  Checks of all cal status values are set to 3 (fully calibrated)
 *  @return status of calibration
 */
bool RPI_BNO055::isFullyCalibrated() {
  uint8_t system, gyro, accel, mag;
  getCalibration(&system, &gyro, &accel, &mag);
  
  switch (_mode) {
    case OPERATION_MODE_ACCONLY:
      return (accel == 3);
    case OPERATION_MODE_MAGONLY:
      return (mag == 3);
    case OPERATION_MODE_GYRONLY:
    case OPERATION_MODE_M4G:
      return (gyro == 3);
    case OPERATION_MODE_ACCMAG:
    case OPERATION_MODE_COMPASS:
      return (accel == 3 && mag == 3);
    case OPERATION_MODE_ACCGYRO:
    case OPERATION_MODE_IMUPLUS:
      return (accel == 3 && gyro == 3);
    case OPERATION_MODE_MAGGYRO:
      return (mag == 3 && gyro == 3);
    default:
      return (system == 3 && gyro == 3 && accel == 3 && mag == 3);
  }
}

// I2C Read/Write Functions using pigpio

/*!
 *  @brief  Writes an 8 bit value over I2C
 */
bool RPI_BNO055::write8(bno055_reg_t reg, uint8_t value) {
  // Ensure pigpio is initialized
  if (!initPigpio()) {
    return false;
  }
  
  char buffer[2];
  buffer[0] = (char)reg;
  buffer[1] = (char)value;
  
  int result = i2cWriteDevice(_i2cHandle, buffer, 2);
  return (result >= 0);
}

/*!
 *  @brief  Reads an 8 bit value over I2C
 */
uint8_t RPI_BNO055::read8(bno055_reg_t reg) {
  // Ensure pigpio is initialized
  if (!initPigpio()) {
    return 0;
  }
  
  char regAddr = (char)reg;
  i2cWriteDevice(_i2cHandle, &regAddr, 1);
  
  char value;
  i2cReadDevice(_i2cHandle, &value, 1);
  
  return (uint8_t)value;
}

/*!
 *  @brief  Reads the specified number of bytes over I2C
 */
bool RPI_BNO055::readLen(bno055_reg_t reg, uint8_t *buffer, uint8_t len) {
  // Ensure pigpio is initialized
  if (!initPigpio()) {
    return false;
  }
  
  char regAddr = (char)reg;
  i2cWriteDevice(_i2cHandle, &regAddr, 1);
  
  int result = i2cReadDevice(_i2cHandle, (char*)buffer, len);
  return (result >= 0);
}

/*!
 *  @brief  Enter Suspend mode (i.e., sleep)
 */
void RPI_BNO055::enterSuspendMode() {
  bno055_opmode_t modeback = _mode;
  
  // Switch to config mode
  setMode(OPERATION_MODE_CONFIG);
  delay(25);
  
  write8(BNO055_PWR_MODE_ADDR, 0x02);
  
  // Return to previous mode
  setMode(modeback);
  delay(20);
}

/*!
 *  @brief  Enter Normal mode (i.e., wake)
 */
void RPI_BNO055::enterNormalMode() {
  bno055_opmode_t modeback = _mode;
  
  // Switch to config mode
  setMode(OPERATION_MODE_CONFIG);
  delay(25);
  
  write8(BNO055_PWR_MODE_ADDR, 0x00);
  
  // Return to previous mode
  setMode(modeback);
  delay(20);
}

/*!
 *  @brief  Gets the sensor's revision information
 *  @param  info
 *          Pointer to a bno055_rev_info_t structure to hold the data
 */
void RPI_BNO055::getRevInfo(bno055_rev_info_t *info) {
  memset(info, 0, sizeof(bno055_rev_info_t));
  
  // Check the accelerometer revision
  info->accel_rev = read8(BNO055_ACCEL_REV_ID_ADDR);
  
  // Check the magnetometer revision
  info->mag_rev = read8(BNO055_MAG_REV_ID_ADDR);
  
  // Check the gyroscope revision
  info->gyro_rev = read8(BNO055_GYRO_REV_ID_ADDR);
  
  // Check the SW revision
  info->bl_rev = read8(BNO055_BL_REV_ID_ADDR);
  
  // Read SW revision
  uint8_t a = read8(BNO055_SW_REV_ID_LSB_ADDR);
  uint8_t b = read8(BNO055_SW_REV_ID_MSB_ADDR);
  info->sw_rev = (((uint16_t)b) << 8) | ((uint16_t)a);
}

/*!
 *  @brief  Gets current calibration data from the sensor
 *  @param  calibData
 *          Calibration data storage buffer (must be at least 22 bytes)
 *  @return true if successful
 */
bool RPI_BNO055::getSensorOffsets(uint8_t *calibData) {
  if (isFullyCalibrated()) {
    bno055_opmode_t lastMode = _mode;
    setMode(OPERATION_MODE_CONFIG);
    delay(25);
    
    // Read the 22 bytes of calibration data
    readLen(ACCEL_OFFSET_X_LSB_ADDR, calibData, NUM_BNO055_OFFSET_REGISTERS);
    
    setMode(lastMode);
    return true;
  }
  return false;
}

/*!
 *  @brief  Gets sensor offsets in offset struct format
 *  @param  offsets_type
 *          Offsets structure to fill with data
 *  @return true if successful
 */
bool RPI_BNO055::getSensorOffsets(bno055_offsets_t &offsets_type) {
  if (isFullyCalibrated()) {
    bno055_opmode_t lastMode = _mode;
    setMode(OPERATION_MODE_CONFIG);
    delay(25);
    
    // Read acceleration offset values
    offsets_type.accel_offset_x = (read8(ACCEL_OFFSET_X_MSB_ADDR) << 8) | 
                                  (read8(ACCEL_OFFSET_X_LSB_ADDR));
    offsets_type.accel_offset_y = (read8(ACCEL_OFFSET_Y_MSB_ADDR) << 8) | 
                                  (read8(ACCEL_OFFSET_Y_LSB_ADDR));
    offsets_type.accel_offset_z = (read8(ACCEL_OFFSET_Z_MSB_ADDR) << 8) | 
                                  (read8(ACCEL_OFFSET_Z_LSB_ADDR));
    
    // Read magnetometer offset values
    offsets_type.mag_offset_x = (read8(MAG_OFFSET_X_MSB_ADDR) << 8) | 
                                (read8(MAG_OFFSET_X_LSB_ADDR));
    offsets_type.mag_offset_y = (read8(MAG_OFFSET_Y_MSB_ADDR) << 8) | 
                                (read8(MAG_OFFSET_Y_LSB_ADDR));
    offsets_type.mag_offset_z = (read8(MAG_OFFSET_Z_MSB_ADDR) << 8) | 
                                (read8(MAG_OFFSET_Z_LSB_ADDR));
    
    // Read gyroscope offset values
    offsets_type.gyro_offset_x = (read8(GYRO_OFFSET_X_MSB_ADDR) << 8) | 
                                (read8(GYRO_OFFSET_X_LSB_ADDR));
    offsets_type.gyro_offset_y = (read8(GYRO_OFFSET_Y_MSB_ADDR) << 8) | 
                                (read8(GYRO_OFFSET_Y_LSB_ADDR));
    offsets_type.gyro_offset_z = (read8(GYRO_OFFSET_Z_MSB_ADDR) << 8) | 
                                (read8(GYRO_OFFSET_Z_LSB_ADDR));
    
    // Read radius values
    offsets_type.accel_radius = (read8(ACCEL_RADIUS_MSB_ADDR) << 8) | 
                              (read8(ACCEL_RADIUS_LSB_ADDR));
    offsets_type.mag_radius = (read8(MAG_RADIUS_MSB_ADDR) << 8) | 
                            (read8(MAG_RADIUS_LSB_ADDR));
    
    setMode(lastMode);
    return true;
  }
  return false;
}

/*!
 *  @brief  Sets calibration data to the sensor
 *  @param  calibData
 *          Calibration data (22 bytes)
 */
void RPI_BNO055::setSensorOffsets(const uint8_t *calibData) {
  bno055_opmode_t lastMode = _mode;
  setMode(OPERATION_MODE_CONFIG);
  delay(25);
  
  // Write the calibration data one byte at a time
  for (int i = 0; i < NUM_BNO055_OFFSET_REGISTERS; i++) {
    write8((bno055_reg_t)(ACCEL_OFFSET_X_LSB_ADDR + i), calibData[i]);
    delay(1);
  }
  
  setMode(lastMode);
  delay(10);
}

/*!
 *  @brief  Sets sensor offsets from an offset struct
 *  @param  offsets_type
 *          Offsets to set
 */
void RPI_BNO055::setSensorOffsets(const bno055_offsets_t &offsets_type) {
  bno055_opmode_t lastMode = _mode;
  setMode(OPERATION_MODE_CONFIG);
  delay(25);
  
  // Write acceleration offset values
  write8(ACCEL_OFFSET_X_LSB_ADDR, (offsets_type.accel_offset_x) & 0xFF);
  write8(ACCEL_OFFSET_X_MSB_ADDR, (offsets_type.accel_offset_x >> 8) & 0xFF);
  write8(ACCEL_OFFSET_Y_LSB_ADDR, (offsets_type.accel_offset_y) & 0xFF);
  write8(ACCEL_OFFSET_Y_MSB_ADDR, (offsets_type.accel_offset_y >> 8) & 0xFF);
  write8(ACCEL_OFFSET_Z_LSB_ADDR, (offsets_type.accel_offset_z) & 0xFF);
  write8(ACCEL_OFFSET_Z_MSB_ADDR, (offsets_type.accel_offset_z >> 8) & 0xFF);
  
  // Write magnetometer offset values
  write8(MAG_OFFSET_X_LSB_ADDR, (offsets_type.mag_offset_x) & 0xFF);
  write8(MAG_OFFSET_X_MSB_ADDR, (offsets_type.mag_offset_x >> 8) & 0xFF);
  write8(MAG_OFFSET_Y_LSB_ADDR, (offsets_type.mag_offset_y) & 0xFF);
  write8(MAG_OFFSET_Y_MSB_ADDR, (offsets_type.mag_offset_y >> 8) & 0xFF);
  write8(MAG_OFFSET_Z_LSB_ADDR, (offsets_type.mag_offset_z) & 0xFF);
  write8(MAG_OFFSET_Z_MSB_ADDR, (offsets_type.mag_offset_z >> 8) & 0xFF);
  
  // Write gyroscope offset values
  write8(GYRO_OFFSET_X_LSB_ADDR, (offsets_type.gyro_offset_x) & 0xFF);
  write8(GYRO_OFFSET_X_MSB_ADDR, (offsets_type.gyro_offset_x >> 8) & 0xFF);
  write8(GYRO_OFFSET_Y_LSB_ADDR, (offsets_type.gyro_offset_y) & 0xFF);
  write8(GYRO_OFFSET_Y_MSB_ADDR, (offsets_type.gyro_offset_y >> 8) & 0xFF);
  write8(GYRO_OFFSET_Z_LSB_ADDR, (offsets_type.gyro_offset_z) & 0xFF);
  write8(GYRO_OFFSET_Z_MSB_ADDR, (offsets_type.gyro_offset_z >> 8) & 0xFF);
  
  // Write radius values
  write8(ACCEL_RADIUS_LSB_ADDR, (offsets_type.accel_radius) & 0xFF);
  write8(ACCEL_RADIUS_MSB_ADDR, (offsets_type.accel_radius >> 8) & 0xFF);
  write8(MAG_RADIUS_LSB_ADDR, (offsets_type.mag_radius) & 0xFF);
  write8(MAG_RADIUS_MSB_ADDR, (offsets_type.mag_radius >> 8) & 0xFF);
  
  setMode(lastMode);
  delay(10);
}

/*!
 *  @brief  Sets axis remap
 *  @param  remapcode
 */
void RPI_BNO055::setAxisRemap(bno055_axis_remap_config_t remapcode) {
  bno055_opmode_t modeback = _mode;
  
  setMode(OPERATION_MODE_CONFIG);
  delay(25);
  
  write8(BNO055_AXIS_MAP_CONFIG_ADDR, remapcode);
  delay(10);
  
  setMode(modeback);
  delay(20);
}

/*!
 *  @brief  Sets axis sign
 *  @param  remapsign
 */
void RPI_BNO055::setAxisSign(bno055_axis_remap_sign_t remapsign) {
  bno055_opmode_t modeback = _mode;
  
  setMode(OPERATION_MODE_CONFIG);
  delay(25);
  
  write8(BNO055_AXIS_MAP_SIGN_ADDR, remapsign);
  delay(10);
  
  setMode(modeback);
  delay(20);
}

/*!
 *  @brief  Gets system status
 *  @param  system_status
 *  @param  self_test_result
 *  @param  system_error
 */
void RPI_BNO055::getSystemStatus(uint8_t *system_status, uint8_t *self_test_result, uint8_t *system_error) {
  write8(BNO055_PAGE_ID_ADDR, 0);
  
  if (system_status != NULL)
    *system_status = read8(BNO055_SYS_STAT_ADDR);
    
  if (self_test_result != NULL)
    *self_test_result = read8(BNO055_SELFTEST_RESULT_ADDR);
    
  if (system_error != NULL)
    *system_error = read8(BNO055_SYS_ERR_ADDR);
    
  delay(200);
}

// Create sensor instance (default address is 0x28)
RPI_BNO055 bno; 
