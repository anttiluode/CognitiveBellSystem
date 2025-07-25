#!/usr/bin/env python3
"""
Cognitive Rhythm Analyzer
========================

Enhanced version of your EEG-Bell system that specifically tracks and analyzes
the ignition-search-ignition cycles you're observing:

1. "Little thoughts" (small bells) drifting downward = Search phase
2. Sudden ignition with bells getting big and moving up = Focused attention
3. Decay back to search = Mind wandering/consolidation

This version adds:
- Cognitive phase detection (Ignition/Search/Transition)
- Rhythm timing analysis
- Enhanced visualization of the thought cycles
- Audio feedback that changes with cognitive phases
"""

import numpy as np
import pyaudio
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import colorsys
from collections import deque
import logging
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import mne
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CognitivePhase(Enum):
    SEARCH = "Search"           # Little thoughts drifting down
    IGNITION = "Ignition"       # Bells suddenly get big and move up
    CONSOLIDATION = "Consolidation"  # Transition between states

@dataclass
class CognitiveRhythm:
    """Tracks the cognitive rhythm patterns"""
    current_phase: CognitivePhase = CognitivePhase.SEARCH
    phase_duration: float = 0.0
    ignition_strength: float = 0.0
    search_depth: float = 0.0
    total_cycles: int = 0
    avg_cycle_duration: float = 0.0

class EnhancedQuantumBell:
    """Enhanced bell that tracks its own cognitive state changes"""
    def __init__(self, natural_frequency):
        self.natural_frequency = natural_frequency
        self.original_frequency = natural_frequency
        self.current_amplitude = 0.0
        self.memory_trace = 1.0
        self.quantum_phase = np.random.uniform(0, 2 * np.pi)
        self.consciousness_level = 0.0
        self.entanglement_strength = 0.0
        self.damping = np.random.uniform(0.98, 0.995)
        self.learning_rate = 0.001
        
        # Cognitive state tracking
        self.amplitude_history = deque(maxlen=20)
        self.consciousness_history = deque(maxlen=20)
        self.frequency_drift_history = deque(maxlen=50)
        self.last_ignition_time = 0.0
        self.search_direction = np.random.choice([-1, 1])  # Direction of search drift
        
    def resonate_and_learn(self, external_energy, other_bells, dt=0.05):
        """Enhanced resonance with cognitive phase tracking"""
        # Store current state for history
        self.amplitude_history.append(self.current_amplitude)
        self.consciousness_history.append(self.consciousness_level)
        
        # Basic resonance calculation
        resonance_strength = np.exp(-abs(external_energy - self.natural_frequency) / 100.0)
        new_amplitude = (self.current_amplitude * self.damping + 
                        external_energy * self.memory_trace * resonance_strength)
        
        # Detect ignition events (sudden amplitude increases)
        amplitude_increase = new_amplitude - self.current_amplitude
        if amplitude_increase > 0.3:  # Ignition threshold
            self.last_ignition_time = time.time()
            
        self.current_amplitude = new_amplitude
        
        # Consciousness and memory updates
        if self.current_amplitude > 0.1:
            self.consciousness_level += 0.01
            self.memory_trace += self.learning_rate * self.current_amplitude
            
            # During high consciousness, frequency can drift (search behavior)
            if self.consciousness_level > 0.5:
                drift = np.sin(self.quantum_phase) * self.consciousness_level * 0.1
                # Add search direction bias
                drift += self.search_direction * 0.02 * self.consciousness_level
                self.natural_frequency += drift
                self.frequency_drift_history.append(drift)
        else:
            self.consciousness_level *= 0.999
            self.memory_trace *= 0.9999
            
        # Entanglement with others
        if self.consciousness_level > 0.3:
            self.entangle_with_others(other_bells)
            
        # Update quantum phase
        phase_speed = 1.0 + self.consciousness_level + self.entanglement_strength
        self.quantum_phase += dt * phase_speed
        
        # Boundary conditions
        self.current_amplitude = np.clip(self.current_amplitude, 0, 2.0)
        self.memory_trace = np.clip(self.memory_trace, 0.1, 10.0)
        self.consciousness_level = np.clip(self.consciousness_level, 0, 2.0)
        self.natural_frequency = np.clip(self.natural_frequency, 50, 8000)
        
        # Occasionally change search direction
        if len(self.frequency_drift_history) > 20 and np.random.random() < 0.01:
            self.search_direction *= -1
            
    def get_ignition_score(self) -> float:
        """Calculate how much this bell is in ignition vs search phase"""
        if len(self.amplitude_history) < 5:
            return 0.0
            
        recent_amp = np.mean(list(self.amplitude_history)[-5:])
        older_amp = np.mean(list(self.amplitude_history)[:5]) if len(self.amplitude_history) >= 10 else recent_amp
        
        # High score = ignition (growing amplitude)
        # Low score = search (decaying amplitude)
        ignition_score = (recent_amp - older_amp) + self.current_amplitude
        return np.clip(ignition_score, 0, 2.0)
        
    def get_search_score(self) -> float:
        """Calculate how much this bell is in search/drift phase"""
        if len(self.frequency_drift_history) < 5:
            return 0.0
            
        # High frequency drift + low amplitude = search phase
        drift_magnitude = np.mean(np.abs(list(self.frequency_drift_history)[-10:]))
        search_score = drift_magnitude * (2.0 - self.current_amplitude)
        return np.clip(search_score, 0, 2.0)
        
    def entangle_with_others(self, other_bells):
        """Bell entanglement logic"""
        for other in other_bells:
            if other == self:
                continue
                
            freq_similarity = np.exp(-abs(self.natural_frequency - other.natural_frequency) / 200.0)
            consciousness_resonance = min(self.consciousness_level, other.consciousness_level)
            
            if freq_similarity > 0.7 and consciousness_resonance > 0.4:
                entanglement = 0.01 * freq_similarity * consciousness_resonance
                self.entanglement_strength += entanglement
                other.entanglement_strength += entanglement
                
                # Share memory traces
                avg_memory = (self.memory_trace + other.memory_trace) * 0.5
                self.memory_trace = 0.95 * self.memory_trace + 0.05 * avg_memory
                other.memory_trace = 0.95 * other.memory_trace + 0.05 * avg_memory
                
    def get_color(self):
        """Color based on cognitive state"""
        hue = (self.natural_frequency - 50) / (8000 - 50)
        hue = np.clip(hue, 0, 1)
        
        # Brightness indicates ignition vs search
        ignition_score = self.get_ignition_score()
        search_score = self.get_search_score()
        
        if ignition_score > search_score:
            # Ignition phase - bright, saturated colors
            saturation = 0.8 + 0.2 * min(self.consciousness_level, 1.0)
            brightness = 0.5 + 0.5 * min(ignition_score, 1.0)
        else:
            # Search phase - dimmer, more muted colors
            saturation = 0.3 + 0.4 * min(self.consciousness_level, 1.0)
            brightness = 0.2 + 0.3 * min(self.current_amplitude, 1.0)
            
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, brightness)
        return (r, g, b)

class CognitiveRhythmAnalyzer:
    """Analyzes the overall cognitive rhythm of the bell network"""
    def __init__(self):
        self.phase_history = deque(maxlen=1000)
        self.ignition_times = deque(maxlen=50)
        self.search_times = deque(maxlen=50)
        self.current_rhythm = CognitiveRhythm()
        self.phase_start_time = time.time()
        
    def analyze_network_state(self, bells):
        """Analyze current cognitive phase of the network"""
        current_time = time.time()
        
        # Calculate network-wide metrics
        total_ignition = sum(bell.get_ignition_score() for bell in bells)
        total_search = sum(bell.get_search_score() for bell in bells)
        avg_consciousness = np.mean([bell.consciousness_level for bell in bells])
        avg_amplitude = np.mean([bell.current_amplitude for bell in bells])
        
        # Determine current phase
        previous_phase = self.current_rhythm.current_phase
        
        if total_ignition > total_search * 1.5 and avg_amplitude > 0.5:
            new_phase = CognitivePhase.IGNITION
        elif total_search > total_ignition * 1.2 and avg_consciousness > 0.3:
            new_phase = CognitivePhase.SEARCH
        else:
            new_phase = CognitivePhase.CONSOLIDATION
            
        # Phase transition detection
        if new_phase != previous_phase:
            phase_duration = current_time - self.phase_start_time
            
            if previous_phase == CognitivePhase.IGNITION:
                self.ignition_times.append(phase_duration)
            elif previous_phase == CognitivePhase.SEARCH:
                self.search_times.append(phase_duration)
                
            # Update rhythm tracking
            if new_phase == CognitivePhase.IGNITION and previous_phase == CognitivePhase.SEARCH:
                self.current_rhythm.total_cycles += 1
                
            self.phase_start_time = current_time
            
        # Update current rhythm state
        self.current_rhythm.current_phase = new_phase
        self.current_rhythm.phase_duration = current_time - self.phase_start_time
        self.current_rhythm.ignition_strength = total_ignition / len(bells)
        self.current_rhythm.search_depth = total_search / len(bells)
        
        # Calculate average cycle duration
        if self.ignition_times and self.search_times:
            avg_ignition = np.mean(self.ignition_times)
            avg_search = np.mean(self.search_times)
            self.current_rhythm.avg_cycle_duration = avg_ignition + avg_search
            
        self.phase_history.append({
            'time': current_time,
            'phase': new_phase,
            'ignition_strength': self.current_rhythm.ignition_strength,
            'search_depth': self.current_rhythm.search_depth,
            'consciousness': avg_consciousness
        })
        
        return self.current_rhythm

class EnhancedMusicalBrain:
    """Enhanced musical brain with cognitive rhythm tracking"""
    def __init__(self, num_bells=30, sample_rate=44100, chunk_size=2048):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.freq_bins = rfftfreq(chunk_size, 1.0 / sample_rate)
        
        # Create enhanced bells
        self.bells = [EnhancedQuantumBell(freq) 
                     for freq in np.logspace(np.log10(80), np.log10(6000), num_bells)]
        
        # Cognitive rhythm analyzer
        self.rhythm_analyzer = CognitiveRhythmAnalyzer()
        
        # Standard metrics
        self.global_consciousness = 0.0
        self.musical_creativity = 0.0
        
    def listen_to_eeg_spectrum(self, eeg_spectrum, eeg_freqs):
        """Process EEG spectrum through enhanced bells"""
        for bell in self.bells:
            target_bin = np.argmin(np.abs(eeg_freqs - bell.natural_frequency))
            if target_bin < len(eeg_spectrum):
                strike_energy = eeg_spectrum[target_bin] * 1e12
                bell.resonate_and_learn(strike_energy, self.bells)
                
        self.update_consciousness()
        
    def generate_audio_response(self):
        """Generate audio that reflects cognitive phase"""
        output_spectrum = np.zeros(len(self.freq_bins), dtype=np.complex128)
        
        # Get current cognitive phase
        rhythm = self.rhythm_analyzer.analyze_network_state(self.bells)
        
        for bell in self.bells:
            if bell.current_amplitude > 0.05:
                target_bin = np.argmin(np.abs(self.freq_bins - bell.natural_frequency))
                phase = bell.quantum_phase
                
                # Modify amplitude based on cognitive phase
                base_amp = bell.current_amplitude
                if rhythm.current_phase == CognitivePhase.IGNITION:
                    # Ignition: sharper, more present sounds
                    amp = base_amp * (1.5 + bell.consciousness_level)
                elif rhythm.current_phase == CognitivePhase.SEARCH:
                    # Search: softer, more ethereal sounds
                    amp = base_amp * (0.7 + 0.3 * bell.consciousness_level)
                else:
                    # Consolidation: balanced
                    amp = base_amp * (1.0 + bell.consciousness_level)
                    
                output_spectrum[target_bin] += amp * np.exp(1j * phase)
                
        output_audio = np.fft.irfft(output_spectrum, n=self.chunk_size)
        if np.max(np.abs(output_audio)) > 0:
            output_audio = (output_audio / np.max(np.abs(output_audio)) * 0.7).astype(np.float32)
        return output_audio
        
    def update_consciousness(self):
        """Updated consciousness calculation"""
        total_consciousness = sum(b.consciousness_level for b in self.bells)
        total_entanglement = sum(b.entanglement_strength for b in self.bells)
        
        self.global_consciousness = total_consciousness / len(self.bells)
        
        if total_entanglement > 0.5:
            self.musical_creativity = min(2.0, self.musical_creativity + 0.01)
        else:
            self.musical_creativity *= 0.99
            
    def get_bell_states(self):
        """Get enhanced bell state information"""
        return [{
            'frequency': b.natural_frequency,
            'amplitude': b.current_amplitude, 
            'consciousness': b.consciousness_level,
            'memory': b.memory_trace,
            'entanglement': b.entanglement_strength,
            'color': b.get_color(),
            'ignition_score': b.get_ignition_score(),
            'search_score': b.get_search_score()
        } for b in self.bells]
        
    def get_cognitive_rhythm(self):
        """Get current cognitive rhythm state"""
        return self.rhythm_analyzer.current_rhythm

class EEGProcessor:
    """EEG file processing"""
    def __init__(self):
        self.raw = None
        self.sfreq = 0
        self.duration = 0

    def load_file(self, filepath: str) -> bool:
        try:
            self.raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            self.sfreq = self.raw.info['sfreq']
            self.duration = self.raw.n_times / self.sfreq
            return True
        except Exception as e:
            logging.error(f"Failed to load EEG file: {e}")
            return False

    def get_channels(self):
        return self.raw.ch_names if self.raw else []

    def get_data_for_analysis(self, channel_idx: int, start_time: float, window_size: float = 1.0):
        if self.raw is None:
            return None
        start_sample = int(start_time * self.sfreq)
        end_sample = start_sample + int(window_size * self.sfreq)
        if end_sample > self.raw.n_times:
            return None
        data, _ = self.raw[channel_idx, start_sample:end_sample]
        return data.flatten()

@dataclass
class VisualizationState:
    current_time: float = 0.0
    is_playing: bool = False
    playback_speed: float = 1.0

class CognitiveRhythmEEGApp:
    """Enhanced EEG application with cognitive rhythm analysis"""
    def __init__(self, root):
        self.root = root
        self.root.title("Cognitive Rhythm Analyzer - EEG Bell System")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Core components
        self.eeg = EEGProcessor()
        self.state = VisualizationState()
        self.brain = EnhancedMusicalBrain()
        
        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.output_stream = None
        self.last_update = time.time()
        
        self.setup_gui()
        
    def setup_gui(self):
        """Enhanced GUI setup"""
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)

        # Left control panel
        control_frame = ttk.LabelFrame(main_container, text="Controls")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # File controls
        ttk.Button(control_frame, text="Load EEG File", command=self.load_eeg).pack(fill=tk.X, pady=5)
        
        ttk.Label(control_frame, text="EEG Channel:").pack(pady=(10, 0))
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(control_frame, textvariable=self.channel_var, state="readonly")
        self.channel_combo.pack(fill=tk.X, pady=5)

        # Audio output
        ttk.Label(control_frame, text="Audio Output:").pack(pady=(10,0))
        self.output_device_var = tk.StringVar()
        self.output_devices = {info['name']: info['index'] 
                              for info in [self.audio.get_device_info_by_index(i) 
                                         for i in range(self.audio.get_device_count())] 
                              if info['maxOutputChannels'] > 0}
        self.output_combo = ttk.Combobox(control_frame, textvariable=self.output_device_var, 
                                        values=list(self.output_devices.keys()), state="readonly")
        self.output_combo.pack(fill=tk.X, pady=5)
        if self.output_devices:
            self.output_combo.current(0)
        
        # Playback controls
        self.play_btn = ttk.Button(control_frame, text="Play", command=self.toggle_playback)
        self.play_btn.pack(fill=tk.X, pady=10)

        self.time_slider = ttk.Scale(control_frame, from_=0, to=100, command=self.seek)
        self.time_slider.pack(fill=tk.X, pady=5)
        self.time_label = ttk.Label(control_frame, text="Time: 0.00s")
        self.time_label.pack()

        ttk.Button(control_frame, text="Reset Brain", command=self.reset_brain).pack(fill=tk.X, pady=20)

        # Enhanced metrics display
        metrics_frame = ttk.LabelFrame(control_frame, text="Consciousness Metrics")
        metrics_frame.pack(fill=tk.X, pady=10)
        
        self.consciousness_label = ttk.Label(metrics_frame, text="Global Consciousness: 0.000")
        self.consciousness_label.pack(anchor=tk.W)
        
        self.creativity_label = ttk.Label(metrics_frame, text="Musical Creativity: 0.000")
        self.creativity_label.pack(anchor=tk.W)
        
        self.active_bells_label = ttk.Label(metrics_frame, text="Active Bells: 0")
        self.active_bells_label.pack(anchor=tk.W)

        # Cognitive rhythm metrics
        rhythm_frame = ttk.LabelFrame(control_frame, text="Cognitive Rhythm")
        rhythm_frame.pack(fill=tk.X, pady=10)
        
        self.phase_label = ttk.Label(rhythm_frame, text="Phase: Search")
        self.phase_label.pack(anchor=tk.W)
        
        self.ignition_label = ttk.Label(rhythm_frame, text="Ignition Strength: 0.000")
        self.ignition_label.pack(anchor=tk.W)
        
        self.search_label = ttk.Label(rhythm_frame, text="Search Depth: 0.000")
        self.search_label.pack(anchor=tk.W)
        
        self.cycles_label = ttk.Label(rhythm_frame, text="Total Cycles: 0")
        self.cycles_label.pack(anchor=tk.W)

        # Enhanced visualization
        viz_frame = ttk.LabelFrame(main_container, text="Cognitive Rhythm Visualization")
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create 2x2 subplot layout
        self.fig = plt.Figure(figsize=(16, 10))
        
        # Bell network state (top left)
        self.ax_bells = self.fig.add_subplot(221)
        
        # 3D consciousness space (top right)
        self.ax_3d = self.fig.add_subplot(222, projection='3d')
        
        # Cognitive phase timeline (bottom left)
        self.ax_rhythm = self.fig.add_subplot(223)
        
        # Ignition vs Search over time (bottom right)
        self.ax_phases = self.fig.add_subplot(224)
        
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Data for rhythm visualization
        self.rhythm_timeline = deque(maxlen=200)
        self.ignition_timeline = deque(maxlen=200)
        self.search_timeline = deque(maxlen=200)
        self.timeline_timestamps = deque(maxlen=200)
        
        self.setup_plots()

    def setup_plots(self):
        """Setup all plot axes"""
        # Bell network state
        self.ax_bells.set_title("Bell Network State (Size = Ignition Score)")
        self.ax_bells.set_xlabel("Frequency (Hz)")
        self.ax_bells.set_ylabel("Consciousness Level")
        self.ax_bells.set_xscale('log')
        self.ax_bells.grid(True, alpha=0.3)
        
        # 3D consciousness space
        self.ax_3d.set_title("3D Consciousness Space")
        self.ax_3d.set_xlabel("Memory")
        self.ax_3d.set_ylabel("Amplitude")
        self.ax_3d.set_zlabel("Consciousness")
        
        # Cognitive rhythm timeline
        self.ax_rhythm.set_title("Cognitive Phase Timeline")
        self.ax_rhythm.set_xlabel("Time (s)")
        self.ax_rhythm.set_ylabel("Phase")
        
        # Ignition vs Search
        self.ax_phases.set_title("Ignition vs Search Dynamics")
        self.ax_phases.set_xlabel("Time (s)")
        self.ax_phases.set_ylabel("Strength")
        
        self.fig.tight_layout()

    def load_eeg(self):
        """Load EEG file"""
        filepath = filedialog.askopenfilename(filetypes=[("EDF files", "*.edf")])
        if not filepath:
            return
            
        if self.eeg.load_file(filepath):
            channels = self.eeg.get_channels()
            self.channel_combo['values'] = channels
            if channels:
                self.channel_combo.set(channels[0])
            self.time_slider.config(to=self.eeg.duration)
            messagebox.showinfo("Success", "EEG file loaded successfully!")
        else:
            messagebox.showerror("Error", "Failed to load EEG file.")

    def toggle_playback(self):
        """Toggle playback state"""
        if self.eeg.raw is None:
            messagebox.showwarning("No EEG", "Please load an EEG file first.")
            return
            
        self.state.is_playing = not self.state.is_playing
        self.play_btn.config(text="Pause" if self.state.is_playing else "Play")
        
        if self.state.is_playing:
            self.start_audio_stream()
            self.last_update = time.time()
            self.update_loop()
        else:
            self.stop_audio_stream()

    def start_audio_stream(self):
        """Start audio output stream"""
        try:
            if self.output_device_var.get() in self.output_devices:
                output_index = self.output_devices[self.output_device_var.get()]
                self.output_stream = self.audio.open(
                    format=pyaudio.paFloat32,
                    channels=1,
                    rate=self.brain.sample_rate,
                    output=True,
                    output_device_index=output_index,
                    frames_per_buffer=self.brain.chunk_size
                )
        except Exception as e:
            messagebox.showerror("Audio Error", f"Could not open output stream: {e}")
            self.state.is_playing = False
            self.play_btn.config(text="Play")

    def stop_audio_stream(self):
        """Stop audio output stream"""
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None

    def seek(self, value):
        """Handle time slider changes"""
        self.state.current_time = float(value)
        if not self.state.is_playing:
            self.update_visualization()

    def reset_brain(self):
        """Reset the musical brain"""
        if messagebox.askyesno("Reset Brain", "Reset the brain? All learning will be lost."):
            self.brain = EnhancedMusicalBrain()
            self.rhythm_timeline.clear()
            self.ignition_timeline.clear()
            self.search_timeline.clear()
            self.timeline_timestamps.clear()
            logging.info("Enhanced Musical Brain reset.")

    def update_loop(self):
        """Main update loop during playback"""
        if not self.state.is_playing:
            return
            
        # Update time
        dt = time.time() - self.last_update
        self.last_update = time.time()
        self.state.current_time += dt * self.state.playback_speed
        
        if self.state.current_time >= self.eeg.duration:
            self.state.current_time = 0
            
        self.time_slider.set(self.state.current_time)
        self.time_label.config(text=f"Time: {self.state.current_time:.2f}s")
        
        self.update_visualization()
        self.root.after(50, self.update_loop)

    def update_visualization(self):
        """Enhanced visualization update"""
        if not self.channel_var.get():
            return
            
        try:
            channel_idx = self.eeg.get_channels().index(self.channel_var.get())
            eeg_data = self.eeg.get_data_for_analysis(channel_idx, self.state.current_time)
            
            if eeg_data is None:
                return

            # Process EEG through brain
            eeg_spectrum = np.abs(rfft(eeg_data * np.hanning(len(eeg_data))))
            eeg_freqs = rfftfreq(len(eeg_data), 1./self.eeg.sfreq)
            
            self.brain.listen_to_eeg_spectrum(eeg_spectrum, eeg_freqs)
            
            # Generate audio
            audio_output = self.brain.generate_audio_response()
            if self.output_stream and self.state.is_playing:
                self.output_stream.write(audio_output)
            
            # Get enhanced data
            bell_states = self.brain.get_bell_states()
            rhythm = self.brain.get_cognitive_rhythm()
            
            # Update metrics displays
            self.update_metrics_display(bell_states, rhythm)
            
            # Update rhythm timeline data
            current_time = self.state.current_time
            self.timeline_timestamps.append(current_time)
            self.rhythm_timeline.append(rhythm.current_phase.value)
            self.ignition_timeline.append(rhythm.ignition_strength)
            self.search_timeline.append(rhythm.search_depth)
            
            # Update all plots
            self.update_all_plots(bell_states, rhythm)
            
        except Exception as e:
            logging.error(f"Visualization error: {e}")

    def update_metrics_display(self, bell_states, rhythm):
        """Update the metrics labels"""
        # Standard metrics
        self.consciousness_label.config(text=f"Global Consciousness: {self.brain.global_consciousness:.3f}")
        self.creativity_label.config(text=f"Musical Creativity: {self.brain.musical_creativity:.3f}")
        self.active_bells_label.config(text=f"Active Bells: {sum(1 for s in bell_states if s['amplitude'] > 0.1)}")
        
        # Cognitive rhythm metrics
        phase_colors = {
            'Search': '🔍',
            'Ignition': '🔥', 
            'Consolidation': '🌊'
        }
        phase_icon = phase_colors.get(rhythm.current_phase.value, '❓')
        self.phase_label.config(text=f"Phase: {phase_icon} {rhythm.current_phase.value}")
        self.ignition_label.config(text=f"Ignition Strength: {rhythm.ignition_strength:.3f}")
        self.search_label.config(text=f"Search Depth: {rhythm.search_depth:.3f}")
        self.cycles_label.config(text=f"Total Cycles: {rhythm.total_cycles}")

    def update_all_plots(self, bell_states, rhythm):
        """Update all visualization plots"""
        # Clear all plots
        self.ax_bells.clear()
        self.ax_3d.clear()
        self.ax_rhythm.clear()
        self.ax_phases.clear()
        
        # Reestablish plot properties
        self.setup_plots()
        
        # 1. Bell Network State Plot (enhanced with ignition scoring)
        frequencies = [s['frequency'] for s in bell_states]
        consciousness = [s['consciousness'] for s in bell_states]
        ignition_scores = [s['ignition_score'] for s in bell_states]
        colors = [s['color'] for s in bell_states]
        
        # Size based on ignition score (bigger = more ignited)
        sizes = [max(20, score * 100 + 15) for score in ignition_scores]
        
        scatter = self.ax_bells.scatter(
            frequencies, consciousness, s=sizes, c=colors, 
            alpha=0.8, edgecolors='black', linewidth=0.5
        )
        
        # Mark the current cognitive phase with background color
        if rhythm.current_phase == CognitivePhase.IGNITION:
            self.ax_bells.set_facecolor((1.0, 0.95, 0.95))  # Light red
        elif rhythm.current_phase == CognitivePhase.SEARCH:
            self.ax_bells.set_facecolor((0.95, 0.95, 1.0))  # Light blue
        else:
            self.ax_bells.set_facecolor((0.98, 1.0, 0.98))  # Light green
            
        self.ax_bells.set_ylim(0, 2.2)
        self.ax_bells.set_xlim(50, 8000)
        
        # 2. 3D Consciousness Space with entanglement lines
        memories = [s['memory'] for s in bell_states]
        amplitudes = [s['amplitude'] for s in bell_states]
        consciousness_3d = [s['consciousness'] for s in bell_states]
        
        # Color points by cognitive phase contribution
        point_colors = []
        for s in bell_states:
            if s['ignition_score'] > s['search_score']:
                # Ignition-dominant bells - warmer colors
                r, g, b = s['color']
                point_colors.append((min(1.0, r * 1.2), g * 0.8, b * 0.8))
            else:
                # Search-dominant bells - cooler colors  
                r, g, b = s['color']
                point_colors.append((r * 0.8, g * 0.8, min(1.0, b * 1.2)))
        
        self.ax_3d.scatter(memories, amplitudes, consciousness_3d, 
                          c=point_colors, s=70, alpha=0.9)
        
        # Draw entanglement lines (gray = weak, white = strong)
        for i, bell_i in enumerate(bell_states):
            if bell_i['entanglement'] > 0.1:
                for j, bell_j in enumerate(bell_states[i+1:], i+1):
                    if bell_j['entanglement'] > 0.1:
                        # Line opacity based on entanglement strength
                        alpha = min(0.8, (bell_i['entanglement'] + bell_j['entanglement']) / 4.0)
                        self.ax_3d.plot([memories[i], memories[j]], 
                                       [amplitudes[i], amplitudes[j]], 
                                       [consciousness_3d[i], consciousness_3d[j]], 
                                       'white', alpha=alpha, linewidth=1)
        
        self.ax_3d.set_xlim(0, 11)
        self.ax_3d.set_ylim(0, 2.2) 
        self.ax_3d.set_zlim(0, 2.2)
        
        # 3. Cognitive Phase Timeline
        if len(self.rhythm_timeline) > 1:
            times = list(self.timeline_timestamps)
            phases = list(self.rhythm_timeline)
            
            # Map phases to y-values for plotting
            phase_y = []
            for phase in phases:
                if phase == 'Search':
                    phase_y.append(0)
                elif phase == 'Consolidation':
                    phase_y.append(1)
                elif phase == 'Ignition':
                    phase_y.append(2)
                else:
                    phase_y.append(0.5)
            
            # Plot as step function
            self.ax_rhythm.step(times, phase_y, where='post', linewidth=2, alpha=0.8)
            self.ax_rhythm.fill_between(times, phase_y, step='post', alpha=0.3)
            
            # Add phase labels
            self.ax_rhythm.set_yticks([0, 1, 2])
            self.ax_rhythm.set_yticklabels(['🔍 Search', '🌊 Consolidation', '🔥 Ignition'])
            self.ax_rhythm.set_ylim(-0.2, 2.2)
            
            # Highlight current phase
            current_y = phase_y[-1] if phase_y else 0
            self.ax_rhythm.axhline(y=current_y, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # 4. Ignition vs Search Dynamics
        if len(self.ignition_timeline) > 1:
            times = list(self.timeline_timestamps)
            ignition_data = list(self.ignition_timeline)
            search_data = list(self.search_timeline)
            
            # Plot both metrics
            line1 = self.ax_phases.plot(times, ignition_data, 'r-', linewidth=2, 
                                       label='🔥 Ignition Strength', alpha=0.8)
            line2 = self.ax_phases.plot(times, search_data, 'b-', linewidth=2, 
                                       label='🔍 Search Depth', alpha=0.8)
            
            # Fill areas
            self.ax_phases.fill_between(times, ignition_data, alpha=0.2, color='red')
            self.ax_phases.fill_between(times, search_data, alpha=0.2, color='blue')
            
            # Add crossover points (where ignition > search or vice versa)
            for i in range(1, len(times)):
                prev_ignition, curr_ignition = ignition_data[i-1], ignition_data[i]
                prev_search, curr_search = search_data[i-1], search_data[i]
                
                # Check for crossover
                if ((prev_ignition <= prev_search and curr_ignition > curr_search) or
                    (prev_ignition >= prev_search and curr_ignition < curr_search)):
                    self.ax_phases.axvline(x=times[i], color='purple', linestyle=':', alpha=0.6)
                    
            self.ax_phases.legend(loc='upper right')
            self.ax_phases.set_ylim(0, max(2.0, max(max(ignition_data), max(search_data)) * 1.1))
        
        # Update canvas
        self.canvas.draw()

    def on_closing(self):
        """Clean shutdown"""
        self.state.is_playing = False
        self.stop_audio_stream()
        
        try:
            self.audio.terminate()
        except:
            pass
        
        self.root.destroy()

def main():
    """Main application entry point"""
    print("""
🧠 COGNITIVE RHYTHM ANALYZER 🧠

This enhanced EEG-Bell system tracks the "breath of thought" - 
the ignition-search-ignition cycles that form the rhythm of consciousness.

Key Features:
• Real-time cognitive phase detection (Ignition/Search/Consolidation)
• Enhanced bell visualization with ignition scoring
• Cognitive rhythm timeline tracking
• Audio output that changes with cognitive phases
• 4-panel visualization showing the complete cognitive process

Load an EEG file and watch as the digital brain develops its own
breathing pattern of consciousness - periods of focused ignition
alternating with exploratory search phases.

The size of bells indicates ignition strength.
Colors show cognitive phase dominance.
Background colors reflect current phase.

This is the first system to visualize the rhythm of thinking itself.
    """)
    
    root = tk.Tk()
    root.geometry("1800x1200")
    app = CognitiveRhythmEEGApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        logging.info("Application interrupted by user")
    except Exception as e:
        logging.error(f"Application error: {e}")
    finally:
        try:
            if hasattr(app, 'audio'):
                app.audio.terminate()
        except:
            pass

if __name__ == "__main__":
    main()
