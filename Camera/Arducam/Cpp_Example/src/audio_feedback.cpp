#include <pigpio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>

// Audio parameters
const int AUDIO_PIN_LEFT = 18;  // GPIO pin for left speaker
const int AUDIO_PIN_RIGHT = 19; // GPIO pin for right speaker
const int AUDIO_FREQ = 1000;    // Base frequency in Hz
const int AUDIO_RANGE = 1000;   // PWM range (0-1000)
const int BASE_VOLUME = 200;    // Base volume level (out of 1000)

// Gap structure (copied from preview_depth_IMU.cpp)
struct Gap {
    int start;
    int end;
    int length() const { return end - start + 1; }
};

class AudioFeedback {
private:
    int leftPin;
    int rightPin;
    bool initialized;

public:
    AudioFeedback(int leftPin = AUDIO_PIN_LEFT, int rightPin = AUDIO_PIN_RIGHT)
        : leftPin(leftPin), rightPin(rightPin), initialized(false) {}

    bool initialize() {
        if (gpioInitialise() < 0) {
            std::cerr << "Failed to initialize pigpio" << std::endl;
            return false;
        }

        // Set pins as output
        gpioSetMode(leftPin, PI_OUTPUT);
        gpioSetMode(rightPin, PI_OUTPUT);

        // Set PWM frequency
        gpioSetPWMfrequency(leftPin, AUDIO_FREQ);
        gpioSetPWMfrequency(rightPin, AUDIO_FREQ);

        // Set PWM range
        gpioSetPWMrange(leftPin, AUDIO_RANGE);
        gpioSetPWMrange(rightPin, AUDIO_RANGE);

        initialized = true;
        return true;
    }

    void updateAudio(const std::vector<Gap>& gaps, int imageWidth) {
        if (!initialized) {
            std::cerr << "AudioFeedback not initialized" << std::endl;
            return;
        }

        // If no gaps found, play base volume on both speakers
        if (gaps.empty()) {
            gpioPWM(leftPin, BASE_VOLUME);
            gpioPWM(rightPin, BASE_VOLUME);
            return;
        }

        // Find the largest gap (most likely path)
        const Gap& mainGap = gaps[0];
        
        // Calculate the center of the gap
        float gapCenter = (mainGap.start + mainGap.end) / 2.0f;
        
        // Calculate the center of the image
        float imageCenter = imageWidth / 2.0f;
        
        // Calculate distance from center (normalized to [0, 1])
        float distanceFromCenter = std::abs(gapCenter - imageCenter) / imageCenter;
        
        // Calculate volume based on distance from center
        // The farther from center, the louder the volume
        int maxVolume = AUDIO_RANGE;
        int volume = BASE_VOLUME + static_cast<int>(distanceFromCenter * (maxVolume - BASE_VOLUME));
        
        // Set volumes based on which side of center the gap is
        if (gapCenter < imageCenter) {
            // Gap is on the left side
            gpioPWM(leftPin, volume);
            gpioPWM(rightPin, BASE_VOLUME);
        } else {
            // Gap is on the right side
            gpioPWM(leftPin, BASE_VOLUME);
            gpioPWM(rightPin, volume);
        }
    }

    void stop() {
        if (initialized) {
            gpioPWM(leftPin, 0);
            gpioPWM(rightPin, 0);
            gpioTerminate();
            initialized = false;
        }
    }

    ~AudioFeedback() {
        stop();
    }
};

