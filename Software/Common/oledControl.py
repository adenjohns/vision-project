# SPDX-FileCopyrightText: 2017 Tony DiCola for Adafruit Industries
# SPDX-FileCopyrightText: 2017 James DeVito for Adafruit Industries
# SPDX-License-Identifier: MIT

# This example is for use on (Linux) computers that are using CPython with
# Adafruit Blinka to support CircuitPython libraries. CircuitPython does
# not support PIL/pillow (python imaging library)!

import time
import subprocess

from board import SCL, SDA
import busio
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306
import pigpio

# Attempt to connect to the pigpio daemon
pi = pigpio.pi()

# 8-bit value
x = 1
b0 = 0
b1 = 0
b2 = 0
b3 = 0
b4 = 0
b5 = 0
b6 = 0
b7 = 0

# GPIO Pins
DB0 = 23 # GPIO23, pin 16
DB1 = 24 # GPIO24, pin 18
DB2 = 10 # GPIO10, pin 19
DB3 = 9 # GPIO9, pin 21
DB4 = 25 # GPIO25, pin 22
DB5 = 11 # GPIO11, pin 23
DB6 = 8 # GPIO8, pin 24
DB7 = 7 # GPIO7, pin 26

# Setting each pins as input
pi.set_mode(DB0, pigpio.INPUT)
pi.set_mode(DB1, pigpio.INPUT)
pi.set_mode(DB2, pigpio.INPUT)
pi.set_mode(DB3, pigpio.INPUT)
pi.set_mode(DB4, pigpio.INPUT)
pi.set_mode(DB5, pigpio.INPUT)
pi.set_mode(DB6, pigpio.INPUT)
pi.set_mode(DB7, pigpio.INPUT)

# Create the I2C interface.
i2c = busio.I2C(SCL, SDA)

# Create the SSD1306 OLED class.
# The first two parameters are the pixel width and pixel height.  Change these
# to the right size for your display!
disp = adafruit_ssd1306.SSD1306_I2C(128, 32, i2c)

# Clear display.
disp.fill(0)
disp.show()

# Create blank image for drawing.
# Make sure to create image with mode '1' for 1-bit color.
width = disp.width
height = disp.height
image = Image.new("1", (width, height))

# Get drawing object to draw on image.
draw = ImageDraw.Draw(image)

# Draw a black filled box to clear the image.
draw.rectangle((0, 0, width, height), outline=0, fill=0)

# Draw some shapes.
# First define some constants to allow easy resizing of shapes.
padding = -2
top = padding
bottom = height - padding


# Load default font.
font = ImageFont.load_default()

# Alternatively load a TTF font.  Make sure the .ttf font file is in the
# same directory as the python script!
# Some other nice fonts to try: http://www.dafont.com/bitmap.php
# font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 9)

while True:
    # Draw a black filled box to clear the image.
    draw.rectangle((0, 0, width, height), outline=0, fill=0)

    # Reading each pins
    if pi.read(7) == 1:
        time.sleep(0.001)
        b7 = 1
    if pi.read(8) == 1:
        time.sleep(0.001)
        b6 = 1
    if pi.read(11) == 1:
        time.sleep(0.001)
        b5 = 1
    if pi.read(25) == 1:
        time.sleep(0.001)
        b4 = 1
    if pi.read(9) == 1:
        time.sleep(0.001)
        b3 = 1
    if pi.read(10) == 1:
        time.sleep(0.001)
        b2 = 1
    if pi.read(24) == 1:
        time.sleep(0.001)
        b1 = 1
    if pi.read(23) == 1:
        time.sleep(0.001)
        b0 = 1

    # Setting the 8-bit value
    x = (1*b0) + (2*b1) + (4*b2) + (8*b3) + (16*b4) + (32*b5) + (64*b6) + (128*b7)

    # Printing the Battery Value
    text_str = f"Battery: {x}"
    draw.text((0, top + 0), text_str, font=font, fill=255)

    # Write four lines of text.

    # draw.text((x, top + 0), "IP: " + IP, font=font, fill=255)
    # draw.text((x, top + 8), "CPU load: " + CPU, font=font, fill=255)
    # draw.text((x, top + 16), MemUsage, font=font, fill=255)
    # draw.text((x, top + 25), Disk, font=font, fill=255)

    # Display image.
    disp.image(image)
    disp.show()
    time.sleep(0.1)
