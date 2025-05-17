#pragma once

#include <alsa/asoundlib.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <thread>
#include <mutex>
#include <atomic>

// Forward declaration of Gap structure
struct Gap;

class AudioFeedbackI2S {
private:
    // ALSA sound device
    snd_pcm_t *pcm_handle;
    
    // Audio parameters
    const char *device_name;
    unsigned int channels;
    unsigned int sample_rate;
    unsigned int buffer_size;
    unsigned int period_size;
    
    // Thread for continuous audio generation
    std::thread audio_thread;
    std::mutex data_mutex;
    std::atomic<bool> running;
    
    // Audio tones parameters
    int base_frequency;
    float base_volume;
    
    // Gap data
    float gap_center;
    float image_center;
    bool has_gap;
    
    // Internal methods
    bool setup_alsa();
    void audio_loop();
    void generate_stereo_sine_wave(float *buffer, int num_samples, float left_vol, float right_vol);

public:
    AudioFeedbackI2S(
        const char *device = "hw:0,0",
        unsigned int channels = 2,
        unsigned int sample_rate = 44100,
        unsigned int buffer_size = 4096,
        unsigned int period_size = 1024,
        int base_frequency = 440, // A4 note
        float base_volume = 0.2f  // 20% volume
    );
    
    ~AudioFeedbackI2S();
    
    bool initialize();
    void updateAudio(const std::vector<Gap>& gaps, int imageWidth);
    void stop();
}; 