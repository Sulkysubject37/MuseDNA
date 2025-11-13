import librosa
import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
import os
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

def get_output_path(audio_path, feature_name):
    """
    Generate a standardized path for output files.
    """
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_filename = f"{base_name}_{feature_name}.png"
    return os.path.join('output', output_filename)

def run_discovery(audio_path):
    """
    Discover the dynamical system governing the musical features of an audio file.
    """
    print("Loading audio for discovery...")
    y, sr = librosa.load(audio_path, sr=44100, mono=True)

    # 1. Extract features (MFCCs) as our time series
    print("Extracting MFCC features...")
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
    # Create a time vector
    n_frames = mfccs.shape[1]
    t = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=512)
    X = mfccs.T  # Transpose to get (n_samples, n_features)
    
    # 2. Set up and fit the SINDy model
    print("Fitting SINDy model to discover governing equations...")
    
    # Define the model components
    feature_library = ps.PolynomialLibrary(degree=2)
    optimizer = ps.STLSQ(threshold=0.1, alpha=0.05)
    
    model = ps.SINDy(
        feature_library=feature_library,
        optimizer=optimizer
    )
    
    # Fit the model
    model.fit(X, t=t, feature_names=[f"m{i}" for i in range(X.shape[1])])
    
    print("\n" + "="*60)
    print("Discovered Dynamical System Equations:")
    model.print()
    print("="*60 + "\n")

    # 3. Simulate the discovered system
    print("Simulating the system from discovered equations...")
    x0 = X[0]  # Initial condition
    x_sim = model.simulate(x0, t)

    # 4. Plot and save the results
    print("Generating validation plot...")
    fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    
    # Plot a few of the MFCC coefficients (real vs. simulated)
    for i in range(4):
        axs[i].plot(t, X[:, i], 'k', label='Original Signal', alpha=0.7)
        axs[i].plot(t, x_sim[:, i], 'r--', label='SINDy Model')
        axs[i].set_ylabel(f'MFCC {i+1}')
        if i == 0:
            axs[i].legend()
            axs[i].set_title(f'Dynamical System Validation: {os.path.basename(audio_path)}', fontweight='bold')

    axs[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    
    output_path = get_output_path(audio_path, 'discovery_validation')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Discovery complete. Validation plot saved to '{output_path}'")
