# Hand Gesture Voxel Editor

This project is a real-time voxel drawing system controlled using hand gestures.

## Features
- Real-time hand tracking
- Pinch gesture to draw blocks
- Pinch on block to erase
- Multiple voxel colors
- Bounce animation
- Webcam interaction

## How It Works
1. Webcam detects your hand
2. Hand landmarks are tracked using MediaPipe
3. Pinch gesture creates a voxel block
4. Moving hand draws blocks across the grid
5. Pinching a block deletes it

## Technologies Used
- Python
- OpenCV
- MediaPipe
- NumPy

## How to Run

Install dependencies:

pip install -r requirements.txt

Run the program:

python voxel_editor.py

Press ESC or Q to exit.
