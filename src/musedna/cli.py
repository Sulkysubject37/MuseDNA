import questionary
import os
import random
import string
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Import the core logic from other modules
from .analysis import run_analysis
from .discovery import run_discovery
from .dna_music import encode_dna, decode_dna

# --- Helper Functions ---

def parse_fasta(file_path):
    """
    Parses a FASTA file, returning a single concatenated DNA sequence.
    """
    sequence = []
    with open(file_path, 'r') as f:
        for line in f:
            if not line.startswith('>'):
                sequence.append(line.strip())
    return "".join(sequence)

def print_banner(console):
    """Prints the application banner."""
    banner_text = Text("MuseDNA", style="bold magenta", justify="center")
    sub_text = Text("A Journey into Musical Genetics", style="cyan", justify="center")
    console.print(Panel(Text.assemble(banner_text, "\n", sub_text)))
    console.print()

# --- Action Handlers ---

def handle_encode():
    """Handler for the encoding action."""
    dna_input = questionary.text(
        "Enter a DNA sequence, or the path to a .txt or .fasta file:",
        validate=lambda text: True if len(text) > 0 else "Please enter a value."
    ).ask()

    if dna_input is None: return # User cancelled

    output_file = questionary.text(
        "Enter the path for the output .wav file:",
        default="output/encoded_song.wav",
        validate=lambda text: True if len(text) > 0 else "Please enter a path."
    ).ask()

    if output_file is None: return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    dna_sequence = ""
    if os.path.exists(dna_input):
        print(f"Reading DNA sequence from file: {dna_input}")
        file_ext = os.path.splitext(dna_input)[1].lower()
        if file_ext in ['.fasta', '.fa']:
            dna_sequence = parse_fasta(dna_input)
        else:
            with open(dna_input, 'r') as f:
                dna_sequence = f.read()
    else:
        dna_sequence = dna_input
    
    if not dna_sequence:
        print("Error: Input DNA sequence is empty.")
        return

    encode_dna(dna_sequence, output_file)

def handle_decode():
    """Handler for the decoding action."""
    audio_file = questionary.path(
        "Enter the path to the audio file to decode:",
        validate=lambda path: os.path.exists(path) or "File not found."
    ).ask()

    if audio_file is None: return

    decoded_sequence, status = decode_dna(audio_file)
    
    if decoded_sequence:
        print("\n--- Decoded DNA Sequence ---")
        print(f"{decoded_sequence[:200]}...")
        print("--------------------------")
    
    color = "green"
    if "Error" in status:
        color = "red"
    
    console = Console()
    console.print(Panel(Text(status, style=f"bold {color}")))

def handle_discover():
    """Handler for the discovery action."""
    audio_file = questionary.path(
        "Enter the path to the audio file to analyze:",
        validate=lambda path: os.path.exists(path) or "File not found."
    ).ask()

    if audio_file is None: return
    os.makedirs("output", exist_ok=True)
    run_discovery(audio_file)

def handle_analysis():
    """Handler for the standard analysis action."""
    audio_file = questionary.path(
        "Enter the path to the audio file to analyze:",
        validate=lambda path: os.path.exists(path) or "File not found."
    ).ask()

    if audio_file is None: return
    os.makedirs("output", exist_ok=True)
    run_analysis(audio_file)

def handle_random():
    """Handler for the random generation action."""
    try:
        length_str = questionary.text(
            "Enter the length of the random DNA sequence:",
            default="100",
            validate=lambda val: val.isdigit() or "Please enter a valid number."
        ).ask()
        
        if length_str is None: return
        length = int(length_str)

        output_file = questionary.text(
            "Enter the path for the output .wav file:",
            default="output/random_song.wav",
            validate=lambda text: True if len(text) > 0 else "Please enter a path."
        ).ask()

        if output_file is None: return
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        random_dna = ''.join(random.choices(['A', 'T', 'G', 'C'], k=length))
        print(f"Generated random DNA of length {length}: {random_dna[:50]}...")
        encode_dna(random_dna, output_file)

    except ValueError:
        print("Invalid length provided.")

def main():
    """Main application loop."""
    console = Console()
    print_banner(console)
    
    actions = {
        "Encode DNA to Music": handle_encode,
        "Decode Music to DNA": handle_decode,
        "Discover Musical DNA (SINDy)": handle_discover,
        "Run Standard Analysis": handle_analysis,
        "Generate Random DNA Music": handle_random,
        "Exit": lambda: True # Signal to exit
    }

    while True:
        try:
            action = questionary.select(
                "Select an action:",
                choices=list(actions.keys())
            ).ask()

            if action is None: # User pressed Ctrl+C
                break

            should_exit = actions[action]()
            if should_exit:
                break
            
            questionary.confirm("Action complete. Return to main menu?").ask()
            console.clear()
            print_banner(console)

        except KeyboardInterrupt:
            break
    
    print("\nExiting MuseDNA. Goodbye!\n")


if __name__ == '__main__':
    # Add rich to requirements
    try:
        import rich
    except ImportError:
        print("Rich library not found. Please run: pip install rich")
        exit()
    main()