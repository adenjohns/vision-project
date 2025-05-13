import os
import time
import warnings

import board
import busio
from PIL import Image, ImageDraw, ImageFont

import adafruit_ssd1306
from adafruit_lc709203f import LC709203F
from adafruit_extended_bus import ExtendedI2C as I2C
# ──────────────────────────────────────────────────────────────────────────────
# Suppress Raspberry‑Pi I²C frequency warning
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="I2C frequency is not settable*"
)

# OLED (SSD1306 128×32) on hardware I²C‑1 (GPIO 2/3)
i2c_oled = busio.I2C(board.SCL, board.SDA)
disp = adafruit_ssd1306.SSD1306_I2C(128, 32, i2c_oled)

# Battery gauge (LC709203F) on bit‑banged I²C‑3 (GPIO 22/23)
i2c_batt = I2C(3)
sensor = LC709203F(i2c_batt)

WIDTH, HEIGHT = disp.width, disp.height
image  = Image.new("1", (WIDTH, HEIGHT))
canvas = ImageDraw.Draw(image)
font   = ImageFont.load_default()

# Reserve space for worst‑case text so logo never hides the SI units
SAMPLES = [
    "Voltage: 99.99V",
    "Charge: 100.0%",
    "Temp: -99.9°C",
]
max_text_width = max(canvas.textbbox((0, 0), s, font=font)[2] for s in SAMPLES)

MARGIN = 1  # now only 1 px between text and logo → allows a slightly wider logo

# ──────────────────────────────────────────────────────────────────────────────
# Load Team Vision logo and scale it as large as possible without clipping text
LOGO_PATH = os.path.join(os.path.dirname(__file__), "Team_Vision", "logo.png")
logo = None
try:
    src = Image.open(LOGO_PATH)
except FileNotFoundError:
    src = None

if src is not None:
    # Convert to pure silhouette (every opaque pixel → white)
    if src.mode in ("RGBA", "LA"):
        mask = src.getchannel("A").point(lambda v: 255 if v > 0 else 0, mode="1")
    else:
        mask = src.convert("L").point(lambda v: 255 if v > 0 else 0, mode="1")

    # Compute the **largest** uniform scale that still fits height AND leaves
    # text + 1 px margin.
    scale_h = HEIGHT / mask.height
    scale_w = (WIDTH - max_text_width - MARGIN) / mask.width
    scale   = min(scale_h, scale_w)

    # Apply scaling if it meaningfully changes size (>1% difference)
    if abs(scale - 1.0) > 0.01:
        new_size = (int(mask.width * scale), int(mask.height * scale))
        mask = mask.resize(new_size, Image.NEAREST)

    logo = mask

# ──────────────────────────────────────────────────────────────────────────────
# Main loop
while True:
    canvas.rectangle((0, 0, WIDTH, HEIGHT), outline=0, fill=0)

    if logo is not None:
        image.paste(logo, (WIDTH - logo.width, 0))

    # Read battery data
    voltage     = sensor.cell_voltage       # V
    percent     = sensor.cell_percent       # %
    temperature = sensor.cell_temperature   # °C

    sensor.thermistor_bconstant = 3950
    sensor.thermistor_enable    = True

    # Print statements (unchanged format)
    canvas.text((0, 0),  f"Voltage: {voltage:0.2f}V", font=font, fill=255)
    canvas.text((0, 8),  f"Charge: {percent:0.1f}%", font=font, fill=255)
    canvas.text((0, 16), f"Temp: {temperature:0.1f}°C", font=font, fill=255)

    disp.image(image)
    disp.show()
    time.sleep(0.1)
