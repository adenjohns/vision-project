# SPDX-FileCopyrightText: 2017 Tony DiCola for Adafruit Industries
# SPDX-FileCopyrightText: 2017 James DeVito for Adafruit Industries
# SPDX-License-Identifier: MIT

# SPDX-License-Identifier: MIT
import time, board, busio, warnings
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306
from adafruit_lc709203f import LC709203F
from adafruit_extended_bus import ExtendedI2C as I2C

# (A) Silence the irrelevant warning
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="I2C frequency is not settable*")

# (B) OLED on the hardware bus /dev/i2c-1 (GPIO 2/3)
i2c_oled = busio.I2C(board.SCL, board.SDA)
disp = adafruit_ssd1306.SSD1306_I2C(128, 32, i2c_oled)

# (C) Battery gauge on the bit-banged bus /dev/i2c-3 (GPIO 22/23)
i2c_batt = I2C(3)     # freq=None suppresses the warning too
sensor   = LC709203F(i2c_batt)

# Clear display.
disp.fill(0)
disp.show()

# Prepare drawing surface
width  = disp.width
height = disp.height
image  = Image.new("1", (width, height))
draw   = ImageDraw.Draw(image)
font   = ImageFont.load_default()

padding = -2
top     = padding

while True:
    # clear buffer
    draw.rectangle((0, 0, width, height), outline=0, fill=0)

    # read battery
    voltage = sensor.cell_voltage
    percent = sensor.cell_percent

    # format text
    voltage_text   = f"Voltage: {voltage:0.3f} V"
    percent_text   = f"Charge:  {percent:3.0f}%"

    # draw text
    draw.text((0, top + 0),  voltage_text, font=font, fill=255)
    draw.text((0, top + 8),  percent_text, font=font, fill=255)
    # optional: show bits
    # draw.text((0, top + 16), "Bits: " + "".join(str(x) for x in b), font=font, fill=255)

    # send buffer to the display
    disp.image(image)
    disp.show()
    time.sleep(0.1)
