#ifndef __RPI_BNO055_H__
#define __RPI_BNO055_H__

#include <stdint.h>
#include <string>
#include <vector>
#include <cmath>
#include <pigpio.h>

// Vector and quaternion classes for IMU math
namespace imu {

template <int N> class Vector {
public:
  Vector() {
    for (int i = 0; i < N; i++)
      _data[i] = 0;
  }

  Vector(const Vector<N> &copy) {
    for (int i = 0; i < N; i++)
      _data[i] = copy._data[i];
  }

  Vector(double a, double b, double c) {
    _data[0] = a;
    _data[1] = b;
    _data[2] = c;
  }

  double &operator[](int i) { return _data[i]; }
  double operator[](int i) const { return _data[i]; }

  double x() const { return _data[0]; }
  double y() const { return _data[1]; }
  double z() const { return _data[2]; }

private:
  double _data[N];
};

class Quaternion {
public:
  Quaternion() : _w(1.0), _x(0.0), _y(0.0), _z(0.0) {}
  Quaternion(double w, double x, double y, double z) : _w(w), _x(x), _y(y), _z(z) {}
  
  double w() const { return _w; }
  double x() const { return _x; }
  double y() const { return _y; }
  double z() const { return _z; }

private:
  double _w, _x, _y, _z;
};

} // namespace imu

/** BNO055 Address A **/
#define BNO055_ADDRESS_A (0x28)
/** BNO055 Address B **/
#define BNO055_ADDRESS_B (0x29)
/** BNO055 ID **/
#define BNO055_ID (0xA0)

/** Offsets registers **/
#define NUM_BNO055_OFFSET_REGISTERS (22)

/** Sensor scaling factors **/
#define SENSORS_DPS_TO_RADS (0.017453293)

/** Default I2C bus **/
#define DEFAULT_I2C_BUS 1

/** A structure to represent offsets **/
typedef struct {
  int16_t accel_offset_x; /**< x acceleration offset */
  int16_t accel_offset_y; /**< y acceleration offset */
  int16_t accel_offset_z; /**< z acceleration offset */

  int16_t mag_offset_x; /**< x magnetometer offset */
  int16_t mag_offset_y; /**< y magnetometer offset */
  int16_t mag_offset_z; /**< z magnetometer offset */

  int16_t gyro_offset_x; /**< x gyroscrope offset */
  int16_t gyro_offset_y; /**< y gyroscrope offset */
  int16_t gyro_offset_z; /**< z gyroscrope offset */

  int16_t accel_radius; /**< acceleration radius */

  int16_t mag_radius; /**< magnetometer radius */
} bno055_offsets_t;

/** Operation mode settings **/
typedef enum {
  OPERATION_MODE_CONFIG = 0X00,
  OPERATION_MODE_ACCONLY = 0X01,
  OPERATION_MODE_MAGONLY = 0X02,
  OPERATION_MODE_GYRONLY = 0X03,
  OPERATION_MODE_ACCMAG = 0X04,
  OPERATION_MODE_ACCGYRO = 0X05,
  OPERATION_MODE_MAGGYRO = 0X06,
  OPERATION_MODE_AMG = 0X07,
  OPERATION_MODE_IMUPLUS = 0X08,
  OPERATION_MODE_COMPASS = 0X09,
  OPERATION_MODE_M4G = 0X0A,
  OPERATION_MODE_NDOF_FMC_OFF = 0X0B,
  OPERATION_MODE_NDOF = 0X0C
} bno055_opmode_t;

/** BNO055 power settings */
typedef enum {
  POWER_MODE_NORMAL = 0X00,
  POWER_MODE_LOWPOWER = 0X01,
  POWER_MODE_SUSPEND = 0X02
} bno055_powermode_t;

/** Remap settings **/
typedef enum {
  REMAP_CONFIG_P0 = 0x21,
  REMAP_CONFIG_P1 = 0x24, // default
  REMAP_CONFIG_P2 = 0x24,
  REMAP_CONFIG_P3 = 0x21,
  REMAP_CONFIG_P4 = 0x24,
  REMAP_CONFIG_P5 = 0x21,
  REMAP_CONFIG_P6 = 0x21,
  REMAP_CONFIG_P7 = 0x24
} bno055_axis_remap_config_t;

/** Remap Signs **/
typedef enum {
  REMAP_SIGN_P0 = 0x04,
  REMAP_SIGN_P1 = 0x00, // default
  REMAP_SIGN_P2 = 0x06,
  REMAP_SIGN_P3 = 0x02,
  REMAP_SIGN_P4 = 0x03,
  REMAP_SIGN_P5 = 0x01,
  REMAP_SIGN_P6 = 0x07,
  REMAP_SIGN_P7 = 0x05
} bno055_axis_remap_sign_t;

/** A structure to represent revisions **/
typedef struct {
  uint8_t accel_rev; /**< acceleration rev */
  uint8_t mag_rev;   /**< magnetometer rev */
  uint8_t gyro_rev;  /**< gyroscrope rev */
  uint16_t sw_rev;   /**< SW rev */
  uint8_t bl_rev;    /**< bootloader rev */
} bno055_rev_info_t;

/** Vector type */
typedef enum {
  VECTOR_ACCELEROMETER = 0x08,
  VECTOR_MAGNETOMETER = 0x0E,
  VECTOR_GYROSCOPE = 0x14,
  VECTOR_EULER = 0x1A,
  VECTOR_LINEARACCEL = 0x28,
  VECTOR_GRAVITY = 0x2E
} vector_type_t;

class RPI_BNO055 {
public:
  /** BNO055 Registers **/
  typedef enum {
    /* Page id register definition */
    BNO055_PAGE_ID_ADDR = 0X07,

    /* PAGE0 REGISTER DEFINITION START*/
    BNO055_CHIP_ID_ADDR = 0x00,
    BNO055_ACCEL_REV_ID_ADDR = 0x01,
    BNO055_MAG_REV_ID_ADDR = 0x02,
    BNO055_GYRO_REV_ID_ADDR = 0x03,
    BNO055_SW_REV_ID_LSB_ADDR = 0x04,
    BNO055_SW_REV_ID_MSB_ADDR = 0x05,
    BNO055_BL_REV_ID_ADDR = 0X06,

    /* Accel data register */
    BNO055_ACCEL_DATA_X_LSB_ADDR = 0X08,
    BNO055_ACCEL_DATA_X_MSB_ADDR = 0X09,
    BNO055_ACCEL_DATA_Y_LSB_ADDR = 0X0A,
    BNO055_ACCEL_DATA_Y_MSB_ADDR = 0X0B,
    BNO055_ACCEL_DATA_Z_LSB_ADDR = 0X0C,
    BNO055_ACCEL_DATA_Z_MSB_ADDR = 0X0D,

    /* Mag data register */
    BNO055_MAG_DATA_X_LSB_ADDR = 0X0E,
    BNO055_MAG_DATA_X_MSB_ADDR = 0X0F,
    BNO055_MAG_DATA_Y_LSB_ADDR = 0X10,
    BNO055_MAG_DATA_Y_MSB_ADDR = 0X11,
    BNO055_MAG_DATA_Z_LSB_ADDR = 0X12,
    BNO055_MAG_DATA_Z_MSB_ADDR = 0X13,

    /* Gyro data registers */
    BNO055_GYRO_DATA_X_LSB_ADDR = 0X14,
    BNO055_GYRO_DATA_X_MSB_ADDR = 0X15,
    BNO055_GYRO_DATA_Y_LSB_ADDR = 0X16,
    BNO055_GYRO_DATA_Y_MSB_ADDR = 0X17,
    BNO055_GYRO_DATA_Z_LSB_ADDR = 0X18,
    BNO055_GYRO_DATA_Z_MSB_ADDR = 0X19,

    /* Euler data registers */
    BNO055_EULER_H_LSB_ADDR = 0X1A,
    BNO055_EULER_H_MSB_ADDR = 0X1B,
    BNO055_EULER_R_LSB_ADDR = 0X1C,
    BNO055_EULER_R_MSB_ADDR = 0X1D,
    BNO055_EULER_P_LSB_ADDR = 0X1E,
    BNO055_EULER_P_MSB_ADDR = 0X1F,

    /* Quaternion data registers */
    BNO055_QUATERNION_DATA_W_LSB_ADDR = 0X20,
    BNO055_QUATERNION_DATA_W_MSB_ADDR = 0X21,
    BNO055_QUATERNION_DATA_X_LSB_ADDR = 0X22,
    BNO055_QUATERNION_DATA_X_MSB_ADDR = 0X23,
    BNO055_QUATERNION_DATA_Y_LSB_ADDR = 0X24,
    BNO055_QUATERNION_DATA_Y_MSB_ADDR = 0X25,
    BNO055_QUATERNION_DATA_Z_LSB_ADDR = 0X26,
    BNO055_QUATERNION_DATA_Z_MSB_ADDR = 0X27,

    /* Linear acceleration data registers */
    BNO055_LINEAR_ACCEL_DATA_X_LSB_ADDR = 0X28,
    BNO055_LINEAR_ACCEL_DATA_X_MSB_ADDR = 0X29,
    BNO055_LINEAR_ACCEL_DATA_Y_LSB_ADDR = 0X2A,
    BNO055_LINEAR_ACCEL_DATA_Y_MSB_ADDR = 0X2B,
    BNO055_LINEAR_ACCEL_DATA_Z_LSB_ADDR = 0X2C,
    BNO055_LINEAR_ACCEL_DATA_Z_MSB_ADDR = 0X2D,

    /* Gravity data registers */
    BNO055_GRAVITY_DATA_X_LSB_ADDR = 0X2E,
    BNO055_GRAVITY_DATA_X_MSB_ADDR = 0X2F,
    BNO055_GRAVITY_DATA_Y_LSB_ADDR = 0X30,
    BNO055_GRAVITY_DATA_Y_MSB_ADDR = 0X31,
    BNO055_GRAVITY_DATA_Z_LSB_ADDR = 0X32,
    BNO055_GRAVITY_DATA_Z_MSB_ADDR = 0X33,

    /* Temperature data register */
    BNO055_TEMP_ADDR = 0X34,

    /* Status registers */
    BNO055_CALIB_STAT_ADDR = 0X35,
    BNO055_SELFTEST_RESULT_ADDR = 0X36,
    BNO055_INTR_STAT_ADDR = 0X37,

    BNO055_SYS_CLK_STAT_ADDR = 0X38,
    BNO055_SYS_STAT_ADDR = 0X39,
    BNO055_SYS_ERR_ADDR = 0X3A,

    /* Unit selection register */
    BNO055_UNIT_SEL_ADDR = 0X3B,

    /* Mode registers */
    BNO055_OPR_MODE_ADDR = 0X3D,
    BNO055_PWR_MODE_ADDR = 0X3E,

    BNO055_SYS_TRIGGER_ADDR = 0X3F,
    BNO055_TEMP_SOURCE_ADDR = 0X40,

    /* Axis remap registers */
    BNO055_AXIS_MAP_CONFIG_ADDR = 0X41,
    BNO055_AXIS_MAP_SIGN_ADDR = 0X42,

    /* Accelerometer Offset registers */
    ACCEL_OFFSET_X_LSB_ADDR = 0X55,
    ACCEL_OFFSET_X_MSB_ADDR = 0X56,
    ACCEL_OFFSET_Y_LSB_ADDR = 0X57,
    ACCEL_OFFSET_Y_MSB_ADDR = 0X58,
    ACCEL_OFFSET_Z_LSB_ADDR = 0X59,
    ACCEL_OFFSET_Z_MSB_ADDR = 0X5A,

    /* Magnetometer Offset registers */
    MAG_OFFSET_X_LSB_ADDR = 0X5B,
    MAG_OFFSET_X_MSB_ADDR = 0X5C,
    MAG_OFFSET_Y_LSB_ADDR = 0X5D,
    MAG_OFFSET_Y_MSB_ADDR = 0X5E,
    MAG_OFFSET_Z_LSB_ADDR = 0X5F,
    MAG_OFFSET_Z_MSB_ADDR = 0X60,

    /* Gyroscope Offset register s*/
    GYRO_OFFSET_X_LSB_ADDR = 0X61,
    GYRO_OFFSET_X_MSB_ADDR = 0X62,
    GYRO_OFFSET_Y_LSB_ADDR = 0X63,
    GYRO_OFFSET_Y_MSB_ADDR = 0X64,
    GYRO_OFFSET_Z_LSB_ADDR = 0X65,
    GYRO_OFFSET_Z_MSB_ADDR = 0X66,

    /* Radius registers */
    ACCEL_RADIUS_LSB_ADDR = 0X67,
    ACCEL_RADIUS_MSB_ADDR = 0X68,
    MAG_RADIUS_LSB_ADDR = 0X69,
    MAG_RADIUS_MSB_ADDR = 0X6A
  } bno055_reg_t;

  // Constructor
  RPI_BNO055(int32_t sensorID = -1, uint8_t address = BNO055_ADDRESS_A, int i2cBus = DEFAULT_I2C_BUS);
  
  // Destructor
  ~RPI_BNO055();

  // Initialize the sensor
  bool begin();
  
  // Set operation mode
  void setMode(bno055_opmode_t mode);
  
  // Get operation mode
  bno055_opmode_t getMode();
  
  // Set axis remap
  void setAxisRemap(bno055_axis_remap_config_t remapcode);
  
  // Set axis sign
  void setAxisSign(bno055_axis_remap_sign_t remapsign);
  
  // Use external crystal
  void setExtCrystalUse(bool usextal);
  
  // Get system status
  void getSystemStatus(uint8_t *system_status, uint8_t *self_test_result, uint8_t *system_error);
  
  // Get calibration state
  void getCalibration(uint8_t *sys, uint8_t *gyro, uint8_t *accel, uint8_t *mag);
  
  // Get temperature
  int8_t getTemp();
  
  // Get vector data (accelerometer, magnetometer, gyroscope, etc)
  imu::Vector<3> getVector(vector_type_t vector_type);
  
  // Get quaternion data
  imu::Quaternion getQuat();
  
  // Get sensor revision info
  void getRevInfo(bno055_rev_info_t *info);
  
  // Get sensor offset values
  bool getSensorOffsets(uint8_t *calibData);
  bool getSensorOffsets(bno055_offsets_t &offsets_type);
  
  // Set sensor offset values
  void setSensorOffsets(const uint8_t *calibData);
  void setSensorOffsets(const bno055_offsets_t &offsets_type);
  
  // Check if fully calibrated
  bool isFullyCalibrated();
  
  // Power management
  void enterSuspendMode();
  void enterNormalMode();

private:
  int _i2cHandle;
  int _i2cBus;
  bno055_opmode_t _mode;
  int32_t _sensorID;
  uint8_t _address;
  
  // Read/write functions using pigpio
  uint8_t read8(bno055_reg_t reg);
  bool write8(bno055_reg_t reg, uint8_t value);
  bool readLen(bno055_reg_t reg, uint8_t *buffer, uint8_t len);
  
  // Helper functions
  bool initPigpio();
  void closePigpio();
  void delay(int ms);
};

#endif /* __RPI_BNO055_H__ */ 
