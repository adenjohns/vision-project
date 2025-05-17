#include "audio_feedback_i2s.h"
#include <cstring>
#include <chrono>

// Gap structure (copied from preview_depth_IMU.cpp)
struct Gap {
    int start;
    int end;
    int length() const { return end - start + 1; }
};

AudioFeedbackI2S::AudioFeedbackI2S(
    const char *device,
    unsigned int channels,
    unsigned int sample_rate,
    unsigned int buffer_size,
    unsigned int period_size,
    int base_frequency,
    float base_volume
) : device_name(device),
    channels(channels),
    sample_rate(sample_rate),
    buffer_size(buffer_size),
    period_size(period_size),
    base_frequency(base_frequency),
    base_volume(base_volume),
    pcm_handle(nullptr),
    running(false),
    has_gap(false),
    gap_center(0),
    image_center(0)
{
}

AudioFeedbackI2S::~AudioFeedbackI2S() {
    stop();
}

bool AudioFeedbackI2S::initialize() {
    // Setup ALSA
    if (!setup_alsa()) {
        std::cerr << "Failed to setup ALSA" << std::endl;
        return false;
    }
    
    // Start audio thread
    running = true;
    audio_thread = std::thread(&AudioFeedbackI2S::audio_loop, this);
    
    return true;
}

bool AudioFeedbackI2S::setup_alsa() {
    int err;
    
    // Open PCM device
    if ((err = snd_pcm_open(&pcm_handle, device_name, SND_PCM_STREAM_PLAYBACK, 0)) < 0) {
        std::cerr << "Cannot open audio device: " << device_name << ", " << snd_strerror(err) << std::endl;
        return false;
    }
    
    // Allocate hardware parameters
    snd_pcm_hw_params_t *hw_params;
    snd_pcm_hw_params_alloca(&hw_params);
    
    // Initialize hardware parameters
    if ((err = snd_pcm_hw_params_any(pcm_handle, hw_params)) < 0) {
        std::cerr << "Cannot initialize hardware parameters: " << snd_strerror(err) << std::endl;
        return false;
    }
    
    // Set access type
    if ((err = snd_pcm_hw_params_set_access(pcm_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) {
        std::cerr << "Cannot set access type: " << snd_strerror(err) << std::endl;
        return false;
    }
    
    // Set sample format (32-bit float)
    if ((err = snd_pcm_hw_params_set_format(pcm_handle, hw_params, SND_PCM_FORMAT_FLOAT)) < 0) {
        std::cerr << "Cannot set sample format: " << snd_strerror(err) << std::endl;
        return false;
    }
    
    // Set sample rate
    if ((err = snd_pcm_hw_params_set_rate_near(pcm_handle, hw_params, &sample_rate, 0)) < 0) {
        std::cerr << "Cannot set sample rate: " << snd_strerror(err) << std::endl;
        return false;
    }
    
    // Set number of channels
    if ((err = snd_pcm_hw_params_set_channels(pcm_handle, hw_params, channels)) < 0) {
        std::cerr << "Cannot set channel count: " << snd_strerror(err) << std::endl;
        return false;
    }
    
    // Set periods
    if ((err = snd_pcm_hw_params_set_periods(pcm_handle, hw_params, 4, 0)) < 0) {
        std::cerr << "Cannot set periods: " << snd_strerror(err) << std::endl;
        return false;
    }
    
    // Set buffer size
    if ((err = snd_pcm_hw_params_set_buffer_size_near(pcm_handle, hw_params, &buffer_size)) < 0) {
        std::cerr << "Cannot set buffer size: " << snd_strerror(err) << std::endl;
        return false;
    }
    
    // Apply hardware parameters
    if ((err = snd_pcm_hw_params(pcm_handle, hw_params)) < 0) {
        std::cerr << "Cannot set hardware parameters: " << snd_strerror(err) << std::endl;
        return false;
    }
    
    // Prepare audio interface
    if ((err = snd_pcm_prepare(pcm_handle)) < 0) {
        std::cerr << "Cannot prepare audio interface: " << snd_strerror(err) << std::endl;
        return false;
    }
    
    return true;
}

void AudioFeedbackI2S::generate_stereo_sine_wave(float *buffer, int num_samples, float left_vol, float right_vol) {
    static float phase = 0.0f;
    const float phase_increment = 2.0f * M_PI * base_frequency / sample_rate;
    
    for (int i = 0; i < num_samples; i++) {
        float sample = sin(phase);
        
        // Interleaved stereo format (LRLR...)
        buffer[i * 2] = sample * left_vol;       // Left channel
        buffer[i * 2 + 1] = sample * right_vol;  // Right channel
        
        phase += phase_increment;
        if (phase > 2.0f * M_PI) {
            phase -= 2.0f * M_PI;
        }
    }
}

void AudioFeedbackI2S::audio_loop() {
    const int frames_per_period = period_size / channels;
    float *buffer = new float[period_size];
    
    while (running) {
        float left_vol = base_volume;
        float right_vol = base_volume;
        
        // Get current gap data under lock
        {
            std::lock_guard<std::mutex> lock(data_mutex);
            
            if (has_gap) {
                // Calculate distance from center (normalized to [0, 1])
                float distanceFromCenter = std::abs(gap_center - image_center) / image_center;
                
                // Calculate volume based on distance from center
                // The farther from center, the louder the volume
                float volume_scale = 1.0f + 4.0f * distanceFromCenter; // Boost volume up to 5x
                
                // Set volumes based on which side of center the gap is
                if (gap_center < image_center) {
                    // Gap is on the left side
                    left_vol = base_volume * volume_scale;
                    right_vol = base_volume;
                } else {
                    // Gap is on the right side
                    left_vol = base_volume;
                    right_vol = base_volume * volume_scale;
                }
            }
        }
        
        // Generate audio data
        generate_stereo_sine_wave(buffer, frames_per_period, left_vol, right_vol);
        
        // Write audio data to ALSA
        int err = snd_pcm_writei(pcm_handle, buffer, frames_per_period);
        
        if (err == -EPIPE) {
            // EPIPE means underrun
            snd_pcm_prepare(pcm_handle);
        } else if (err < 0) {
            std::cerr << "Error from writei: " << snd_strerror(err) << std::endl;
        }
        
        // Sleep to reduce CPU usage (adjust as needed)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    
    delete[] buffer;
}

void AudioFeedbackI2S::updateAudio(const std::vector<Gap>& gaps, int imageWidth) {
    std::lock_guard<std::mutex> lock(data_mutex);
    
    // If no gaps found, equal volume on both sides
    if (gaps.empty()) {
        has_gap = false;
        return;
    }
    
    // Find the largest gap (most likely path)
    const Gap& mainGap = gaps[0];
    
    // Calculate the center of the gap
    gap_center = (mainGap.start + mainGap.end) / 2.0f;
    
    // Calculate the center of the image
    image_center = imageWidth / 2.0f;
    
    has_gap = true;
}

void AudioFeedbackI2S::stop() {
    if (running) {
        running = false;
        
        if (audio_thread.joinable()) {
            audio_thread.join();
        }
        
        if (pcm_handle) {
            snd_pcm_close(pcm_handle);
            pcm_handle = nullptr;
        }
    }
} 