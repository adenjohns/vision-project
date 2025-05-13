#!/bin/bash

# Exit on error
set -e

echo "Installing dependencies for I2S audio..."

# Update package lists
sudo apt-get update

# Install ALSA development libraries
sudo apt-get install -y libasound2-dev

# Install additional tools for debugging
sudo apt-get install -y alsa-utils

echo "Configuring the Raspberry Pi for I2S audio..."

# Check if config.txt already has I2S enabled
if ! grep -q "dtparam=i2s=on" /boot/config.txt; then
    echo "Enabling I2S in /boot/config.txt..."
    sudo sh -c 'echo "dtparam=i2s=on" >> /boot/config.txt'
fi

# Check if specific I2S DAC overlay is needed
echo "Do you have a specific I2S DAC? Enter the overlay name or press Enter to skip:"
echo "Common options: hifiberry-dac, hifiberry-dacplus, iqaudio-dac, etc."
read overlay

if [ ! -z "$overlay" ]; then
    # Check if overlay is already in config.txt
    if ! grep -q "dtoverlay=$overlay" /boot/config.txt; then
        echo "Adding dtoverlay=$overlay to /boot/config.txt..."
        sudo sh -c "echo \"dtoverlay=$overlay\" >> /boot/config.txt"
    else
        echo "Overlay already configured in config.txt"
    fi
fi

echo "Installation complete!"
echo "A system reboot is recommended to apply the changes."
echo "Run 'sudo reboot' to reboot the system."

echo "After reboot, you can check the I2S device with:"
echo "aplay -l    # List all sound devices"
echo "speaker-test -D hw:0,0 -c2 -twav    # Test stereo output" 