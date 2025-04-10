# AI_based_chess_coach
# Live Chess Board Detection.
## Preview
![Chess Board Detection](image/Screenshot(57).png)

Future updates will include further modules such as chess piece detection, move evaluation with Stockfish, and natural language commentary generation with an LLM.

## Features Implemented

- **Live Screen Capture using MSS**
  - Captures a specified window region in real time.
  - Uses `pygetwindow` to identify the target Chrome window.
- **Basic Chess Board Detection**
  - Processes the captured frames with OpenCV.
  - (Future: Integrate object detection to precisely detect the chess board region.)

## Prerequisites

1. **Python 3.8+**  
   Ensure you have Python 3.8 or a later version installed.

2. **Dependencies**  
   Install the required libraries using:
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, install the following packages manually:
   ```bash
   pip install mss opencv-python pygetwindow
   ```

3. **Target Application**  
   - Have a chess website (e.g., Chess.com) open in Chrome.
   - Ensure the window title contains a unique keyword (e.g., "Chess.com").

## How to Run

1. **Configure the Target Window**  
   - The code searches for a Chrome window with a title containing `Chess.com`.
   - Adjust the window search string in the code if needed.

2. **Run the Main Script**  
   ```bash
   python main.py
   ```
   - A new window will open displaying the live captured feed with the detected chess board region.

3. **Stop the Capture**  
   - Press the **`q`** key in the OpenCV window to exit.

## Project Structure

```
.
├── main.py               # Main script for live screen capture and chess board detection
├── README.md             # This documentation file
└── requirements.txt      # Python dependencies
```

- **main.py**  
  Contains the code for detecting the target window, capturing the screen using **mss**, and displaying the live feed with OpenCV.

- **requirements.txt**  
  Lists the following packages (versions can be adjusted as needed):
  ```
  mss
  opencv-python
  pygetwindow
  ultralytics
  pil
  torch
  time
  numpy
  ```

## Current Limitations & Future Plans

- **Current Implementation:**
  - Captures the live screen feed using **mss**.
  - Detects (or crops) the chess board region based on a fixed coordinate or basic detection.
  
- **Future Plans:**
  1. **Board Mapping:**  
     Map detected pieces to an 8×8 grid for board state reconstruction.
  2. **Move Evaluation:**  
     Use Stockfish for move evaluation.
  3. **Natural Language Commentary:**  
     Generate human-readable analysis using a fine-tuned LLM.

## Contributing

- **Feedback & Issues:**  
  Feel free to open issues for bugs or suggestions.
  
- **Pull Requests:**  
  Contributions are welcome. Please ensure your code is well-documented.





---

Enjoy live capturing and detecting your chess board in real time! Stay tuned for future updates on full AI-driven chess analysis.
