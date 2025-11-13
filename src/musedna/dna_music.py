import numpy as np
import librosa
from scipy.io.wavfile import write
import galois
import warnings
import math
import os

warnings.filterwarnings('ignore')

# --- Configuration ---
SAMPLE_RATE = 44100
NOTE_DURATION = 0.2
NOTE_VOLUME = 0.5

# --- Reed-Solomon FEC Configuration ---
K = 23  # Number of message symbols in a block
N = 31  # Total number of symbols in a block (message + parity)
# This can correct up to (N-K)/2 = 4 errors per block.
GF = galois.GF(2**5)  # Galois Field GF(32), must be >= N
RS = galois.ReedSolomon(N, K, field=GF)

# --- Mappings ---
# Map DNA bases to integers for FEC processing
DNA_TO_INT = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
INT_TO_DNA = {v: k for k, v in DNA_TO_INT.items()}

# Create a large frequency map for all possible symbols in the Galois Field
# Using a chromatic scale starting from C4 (MIDI note 60)
midi_notes = np.arange(60, 60 + GF.order)
FREQS = librosa.midi_to_hz(midi_notes)
SYMBOL_TO_FREQ = FREQS
INT_TO_FREQ = {i: SYMBOL_TO_FREQ[i] for i in range(GF.order)}
VALID_FREQS = list(INT_TO_FREQ.values())

def get_output_path(audio_path, feature_name):
    """
    Generate a standardized path for output files.
    """
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_filename = f"{base_name}_{feature_name}.png"
    return os.path.join('output', output_filename)

# --- Audio Generation ---
def generate_rich_note(frequency, duration, sample_rate, volume):
    t = np.linspace(0., duration, int(sample_rate * duration), endpoint=False)
    amplitude = np.iinfo(np.int16).max * volume
    data = np.sin(2. * np.pi * frequency * t)
    data += 0.5 * np.sin(2. * np.pi * 2 * frequency * t)
    decay = np.exp(-np.linspace(0, 5, len(t)))
    data *= decay
    data = amplitude * data / np.max(np.abs(data))
    return data.astype(np.int16)

# --- Core FEC Functions ---
def encode_dna(dna_sequence, output_path):
    print("Preparing DNA sequence for FEC encoding...")
    dna_sequence = ''.join(filter(lambda c: c in 'ATGC', dna_sequence.upper()))
    original_length = len(dna_sequence)
    
    if original_length == 0:
        print("Error: No valid DNA bases found.")
        return False

    # 1. Convert DNA to integer symbols
    int_sequence = [DNA_TO_INT[base] for base in dna_sequence]

    # 2. Pad sequence to be a multiple of K
    padding_needed = (K - (original_length % K)) % K
    padded_sequence = int_sequence + [0] * padding_needed
    
    # 3. Encode the data in blocks using Reed-Solomon
    print(f"Encoding {original_length} DNA bases into blocks of {N} symbols...")
    encoded_blocks = []
    for i in range(0, len(padded_sequence), K):
        block = GF(padded_sequence[i:i+K])
        encoded_block = RS.encode(block)
        encoded_blocks.extend(list(encoded_block))

    # 4. Create a header with the original length
    # We'll use 4 symbols to encode a 16-bit integer length
    header = [
        (original_length >> 12) & 0xF,
        (original_length >> 8) & 0xF,
        (original_length >> 4) & 0xF,
        original_length & 0xF,
    ]
    final_symbols = header + encoded_blocks
    
    # 5. Convert all symbols to audio
    print("Generating audio from symbols...")
    audio_segments = [generate_rich_note(INT_TO_FREQ[int(symbol)], NOTE_DURATION, SAMPLE_RATE, NOTE_VOLUME) for symbol in final_symbols]
    full_audio = np.concatenate(audio_segments)
    write(output_path, SAMPLE_RATE, full_audio)
    print(f"Successfully encoded DNA into '{output_path}'")
    return True

def find_closest_symbol(detected_freq):
    if detected_freq is None or np.isnan(detected_freq): return None
    min_index = np.argmin(np.abs(np.array(VALID_FREQS) - detected_freq))
    return min_index # Return the integer symbol

def decode_dna(audio_path, debug=False):
    print(f"Decoding DNA from '{audio_path}' with FEC...")
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        return None, f"Error loading audio file: {e}"

    # 1. Custom Grid-Based Segmentation
    note_length_samples = int(NOTE_DURATION * sr)
    num_notes = len(y) // note_length_samples
    
    detected_symbols = []
    for i in range(num_notes):
        chunk = y[i*note_length_samples:(i+1)*note_length_samples]
        if chunk.size > 0:
            # Analyze the middle 50% of the chunk to avoid boundary noise
            start_offset = chunk.size // 4
            end_offset = chunk.size * 3 // 4
            stable_chunk = chunk[start_offset:end_offset]

            # FFT-based pitch detection
            if stable_chunk.size == 0: continue
            
            fft_result = np.fft.fft(stable_chunk)
            fft_freq = np.fft.fftfreq(len(stable_chunk), 1/sr)
            
            # Find the peak frequency in the positive spectrum
            peak_idx = np.argmax(np.abs(fft_result[:len(fft_result)//2]))
            detected_freq = fft_freq[peak_idx]

            symbol = find_closest_symbol(detected_freq)
            if symbol is not None:
                detected_symbols.append(symbol)

    if len(detected_symbols) < 4:
        return "", "Error: Audio too short to contain a valid header."

    # 2. Decode the header to get original length
    header_symbols = detected_symbols[:4]
    original_length = (header_symbols[0] << 12) + (header_symbols[1] << 8) + (header_symbols[2] << 4) + header_symbols[3]
    
    # 3. Decode the Reed-Solomon blocks
    print(f"Detected {len(detected_symbols)} symbols. Expecting original length: {original_length}")
    body_symbols = GF(detected_symbols[4:])
    decoded_message = []
    total_errors_corrected = 0
    
    num_blocks = math.ceil(len(body_symbols) / N)
    for i in range(num_blocks):
        block = body_symbols[i*N:(i+1)*N]
        if len(block) == 0: continue
        if len(block) < N:
            # Pad if the last block is incomplete
            block = np.concatenate([block, GF.Zeros(N - len(block))])
        
        try:
            decoded_block = RS.decode(block)
            # Manually calculate the number of corrected errors
            n_errors = np.sum(block[:K] != decoded_block)
            total_errors_corrected += n_errors
            decoded_message.extend([int(s) for s in decoded_block])
        except galois.errors.GaloisError as e:
            return "", f"Error: Too many errors to correct in block {i+1}. {e}"
        except Exception as e:
            return "", f"Error during decoding block {i+1}: {e}"

    # 4. Truncate padding and convert back to DNA
    final_int_sequence = decoded_message[:original_length]
    decoded_dna_str = "".join([INT_TO_DNA.get(i, '?') for i in final_int_sequence])

    status = f"Verified ({total_errors_corrected} errors corrected)"
    print(status)
    
    # 5. Generate debug plot if requested
    if debug:
        print("Generating debug plot...")
        # This debug plot is less critical now but can still be useful
        fig, ax = plt.subplots(figsize=(15, 5))
        librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.7)
        ax.set_title(f'Debug Plot: {os.path.basename(audio_path)}')
        ax.set_ylabel('Amplitude')
        plt.tight_layout()
        output_path = get_output_path(audio_path, 'decode_debug_waveform')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Debug waveform plot saved to '{output_path}'")

    return decoded_dna_str, status