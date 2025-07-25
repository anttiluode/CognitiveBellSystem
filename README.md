# EEG Bell Resonance System

(Vibecoded. Odd. Super Experimental) 

Note! It takes a while for the system to start. The bells are trailing behind the eeg as they 
learn to resonate with it. 

![Cognitive Rhythm Analyzer](pic.png)

An experimental system for visualizing EEG signal dynamics through resonance-based modeling.

## Overview

This application processes EEG data through a network of simulated resonators ("bells") that respond to frequency content in the signal. The system visualizes temporal patterns in EEG data and generates corresponding audio feedback.

## What It Does

- **Signal Processing**: Converts EEG time-series data into frequency domain representations
- **Resonance Modeling**: Maps frequency components to a network of adaptive resonators
- **Pattern Visualization**: Displays system dynamics in real-time through multiple visualization panels
- **Audio Generation**: Produces audio output based on resonator states

## Key Features

### Visualization Panels
1. **Bell Network State**: Shows resonator frequency vs. activity level
2. **3D State Space**: Displays memory, amplitude, and activity relationships
3. **Phase Timeline**: Tracks temporal patterns in system behavior
4. **Dynamics Plot**: Shows competing activity patterns over time

### Resonator Properties
Each resonator has:
- Natural frequency (determines EEG frequency response)
- Amplitude (current activation level)
- Memory trace (adaptive sensitivity)
- Activity level (cumulative response measure)

### Adaptive Behavior
- Resonators can shift their natural frequencies based on input patterns
- Network connections form between similarly-tuned resonators
- System exhibits cyclical patterns of activity and rest

## Observed Phenomena

### Activity Cycles
The system exhibits alternating phases:
- **High Activity**: Resonators show increased amplitude and coordination
- **Low Activity**: Distributed, exploratory behavior with frequency drift
- **Transitions**: Periods of reorganization between states

### Pattern Recognition
- Sustained EEG patterns lead to stable resonator configurations
- Transient signals create temporary activations that decay over time
- The system appears to distinguish between different types of input

### Frequency Mapping
- Low frequency EEG content (1-8 Hz) tends to engage low-frequency resonators  
- Higher frequency content (8+ Hz) activates corresponding resonators
- Complex signals create distributed activation patterns

## Technical Implementation

### Requirements
```
numpy
pyaudio
tkinter
scipy
matplotlib
mne
```

### Usage
1. Load an EEG file (EDF format)
2. Select a channel for analysis
3. Choose audio output device
4. Click Play to begin processing
5. Observe the real-time visualizations

### File Format
Supports standard EDF (European Data Format) EEG files with standard channel naming.

## Experimental Observations

### Sleep vs. Wake Patterns
Different EEG states produce distinct system behaviors:
- **Active/Wake EEG**: Rapid cycling between high and low activity phases
- **Sleep/Rest EEG**: Extended low-activity periods with gradual transitions
- **Artifact/Noise**: Chaotic, non-repetitive patterns

### Temporal Dynamics
- Activity cycles range from seconds to minutes depending on input
- Pattern stability correlates with EEG signal coherence
- System "memory" allows recognition of recurring patterns

## Limitations and Considerations

### Interpretive Cautions
- This is an experimental visualization tool, not a medical device
- Patterns observed may reflect signal processing artifacts
- No claims are made about biological accuracy or clinical relevance
- Results should be interpreted as signal dynamics, not physiological states

### Technical Limitations
- Processing is real-time but not sample-accurate
- Audio output may have latency depending on system
- Large EEG files may require significant processing time
- Visualization updates at ~20 Hz regardless of EEG sampling rate

## Research Applications

This system may be useful for:
- Exploring temporal patterns in EEG data
- Developing intuition about signal dynamics
- Creating audio-visual representations of neural data
- Investigating adaptive signal processing approaches

## Future Directions

Potential improvements:
- Quantitative metrics for pattern classification
- Parameter optimization for different EEG types
- Integration with other physiological signals
- Validation against established EEG analysis methods

## Code Structure

- `CognitiveRhythmEEGApp`: Main application class
- `EnhancedQuantumBell`: Individual resonator implementation
- `EnhancedMusicalBrain`: Resonator network management
- `CognitiveRhythmAnalyzer`: Pattern detection and classification
- `EEGProcessor`: EEG file loading and preprocessing

## License

MIT License - See LICENSE file for details

## Disclaimer

This software is for research and educational purposes only. It is not intended for medical diagnosis or treatment. Always consult qualified medical professionals for health-related decisions.
