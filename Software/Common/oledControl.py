# SPDX-FileCopyrightText: 2017 Tony DiCola for Adafruit Industries
# SPDX-FileCopyrightText: 2017 James DeVito for Adafruit Industries
# SPDX-License-Identifier: MIT

import time
import busio
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306
from adafruit_lc709203f import LC709203F
import pigpio


# Set up GPIO reader
pi = pigpio.pi()

pi.set_mode(22, pigpio.ALT5) 
pi.set_mode(23, pigpio.ALT5) 

# Create the I2C interface.
i2c = busio.I2C(board.SCL, board.SDA)

sensor = LC709203F(i2c)  # Using pin 15 and 16 (GPIO22/23) must use ALT5

# Create the SSD1306 OLED display (128×32)
disp = adafruit_ssd1306.SSD1306_I2C(128, 32, i2c) # Using pin 3 and 5 (GPIO2/3)

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
    percent_text   = f"Charge: {percent:3.0f}%"

    # draw text
    draw.text((0, top + 0),  voltage_text, font=font, fill=255)
    draw.text((0, top + 8),  percent_text, font=font, fill=255)
    # optional: show bits
    # draw.text((0, top + 16), "Bits: " + "".join(str(x) for x in b), font=font, fill=255)

    # send buffer to the display
    disp.image(image)
    disp.show()
    time.sleep(0.1)
