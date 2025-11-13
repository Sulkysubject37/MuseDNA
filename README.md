# MuseDNA: A Journey into Musical Genetics

![MuseDNA Banner](https://i.imgur.com/your-banner-image.png) <!-- You can create and upload a banner image -->

MuseDNA is an interactive command-line application that explores the intersection of music, data science, and bioinformatics. It provides a suite of tools to analyze music, discover its underlying mathematical dynamics, and even use it as a medium for storing and retrieving genetic data with a powerful error-correction system.

---

## Features

MuseDNA operates as an interactive menu-driven application. Simply run `MuseDNA` in your terminal to access the following features:

-   **Encode DNA to Music**: Convert a DNA sequence (from a string, `.txt`, or `.fasta` file) into a harmonically rich `.wav` audio file. This process uses a powerful Reed-Solomon Forward Error Correction (FEC) system to ensure data integrity.
-   **Decode Music to DNA**: Decode a previously generated audio file back into its original DNA sequence. The FEC system automatically corrects errors that may have been introduced, ensuring a robust and reliable retrieval of the data.
-   **Discover Musical DNA (SINDy)**: Analyzes the sonic features (MFCCs) of any `.wav` file and uses Sparse Identification of Nonlinear Dynamics (SINDy) to discover the system of differential equations that govern its evolution.
-   **Run Standard Analysis**: Performs a classic Music Information Retrieval (MIR) analysis on any `.wav` file, generating plots for its waveform, spectrogram, chromagram, and MFCCs.
-   **Generate Random DNA Music**: Creates a new piece of music from a randomly generated DNA sequence of a specified length.

## Tech Stack

-   **Python 3.8+**
-   **TUI/CLI**: [Questionary](https://github.com/tmbo/questionary) & [Rich](https://github.com/Textualize/rich)
-   **Audio Processing**: [Librosa](https://librosa.org/)
-   **Error Correction**: [Galois](https://github.com/mhostetter/galois) (for Reed-Solomon codes)
-   **Dynamical Systems**: [PySINDy](https://github.com/dynamicslab/pysindy)
-   **Numerical**: NumPy & SciPy

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd MuseDNA
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the package in editable mode:** This command installs all required dependencies and creates the `MuseDNA` executable command.
    ```bash
    pip install -e .
    ```

---

## Usage

After installation, you can run the application from anywhere as long as the virtual environment is active.

1.  **Activate the virtual environment:**
    ```bash
    source /path/to/your/MuseDNA/venv/bin/activate
    ```

2.  **Run the application:**
    ```bash
    MuseDNA
    ```

This will launch the interactive menu. Follow the on-screen prompts to use the various features. All output files (audio and plots) will be saved to the `output/` directory by default.
