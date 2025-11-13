import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_audio(file_path):
    """
    Load the audio file.
    """
    y, sr = librosa.load(file_path, sr=44100, mono=True)
    return y, sr

def get_output_path(audio_path, feature_name):
    """
    Generate a standardized path for output files.
    """
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_filename = f"{base_name}_{feature_name}.png"
    return os.path.join('output', output_filename)

def plot_waveform(y, sr, audio_path):
    """
    Plot the time-domain waveform and save it.
    """
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.7)
    plt.title(f'Waveform: {os.path.basename(audio_path)}', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    output_path = get_output_path(audio_path, 'waveform')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path

def plot_spectrogram(y, sr, audio_path, n_fft=2048, hop_length=512):
    """
    Create and save a detailed spectrogram visualization.
    """
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    img = librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length, 
                                   x_axis='time', y_axis='log', ax=ax)
    ax.set_title(f'Spectrogram: {os.path.basename(audio_path)}', fontsize=16, fontweight='bold')
    ax.set_ylabel('Frequency (Hz)')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    
    plt.tight_layout()
    output_path = get_output_path(audio_path, 'spectrogram')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path

def plot_chromagram(y, sr, audio_path, hop_length=512):
    """
    Create and save a chromagram for harmonic analysis.
    """
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(chroma, sr=sr, hop_length=hop_length, 
                            x_axis='time', y_axis='chroma', 
                            cmap='coolwarm')
    plt.title(f'Chromagram: {os.path.basename(audio_path)}', fontsize=16, fontweight='bold')
    plt.colorbar()
    plt.tight_layout()
    output_path = get_output_path(audio_path, 'chromagram')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path

def plot_mfcc(y, sr, audio_path, hop_length=512):
    """
    Plot and save MFCCs for timbral analysis.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(mfccs, sr=sr, hop_length=hop_length, 
                            x_axis='time', cmap='viridis')
    plt.title(f'MFCCs: {os.path.basename(audio_path)}', fontsize=16, fontweight='bold')
    plt.colorbar()
    plt.ylabel('MFCC Coefficients')
    plt.tight_layout()
    output_path = get_output_path(audio_path, 'mfcc')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path

def run_analysis(audio_path):
    """
    Main function to run all standard analyses.
    """
    print("Loading audio...")
    y, sr = load_audio(audio_path)
    
    print("Generating waveform plot...")
    plot_waveform(y, sr, audio_path)
    
    print("Generating spectrogram...")
    plot_spectrogram(y, sr, audio_path)
    
    print("Generating chromagram...")
    plot_chromagram(y, sr, audio_path)
    
    print("Generating MFCC plot...")
    plot_mfcc(y, sr, audio_path)
    
    print(f"\nAnalysis complete. All plots saved in the 'output/' directory.")
