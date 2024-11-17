# SegmentBot: AI-Powered Image Segmentation and Editing Tool

SegmentBot is a web-based application for performing zero-shot segmentation and image editing using state-of-the-art AI models such as GroundingDINO and Segment Anything Model (SAM). It provides an intuitive interface for detecting objects in images using natural language prompts and editing them with customizable effects like blurring.
![UI](https://github.com/SvenPfiffner/SegmentBot/blob/main/ReadmeMedia/UI.png)

## Features
- Zero-shot Segmentation: Detect objects in images using textual prompts without pre-training on specific classes.
- Editing Modes:
  - Retrieve the binary mask of the detection
  - Blur specific objects.
  - Inverse blur (blur everything except specified objects).
  - ...
- Streamlit-based Interface:
  - Upload images, configure settings, and view/download processed results.
- Extensibility: Add your own editors using a modular system.

### Example uses
Seamlessly blur the background of subjects by prompting what to keep unblurred (In the example: Human)
![Blur](https://github.com/SvenPfiffner/SegmentBot/blob/main/ReadmeMedia/Backgroundblur.png)

Ensure privacy by prompting to segment human faces and setting the edit mode to pixelation
![Pixel](https://github.com/SvenPfiffner/SegmentBot/blob/main/ReadmeMedia/FacePixels.png)

Create Binary masks of objects by prompting for the desired object and setting the edit mode to return the raw mask
![Mask](https://github.com/SvenPfiffner/SegmentBot/blob/main/ReadmeMedia/CarMask.png)

## Extending the Application
Add a New Editor:
1. Create a Python file in the editors directory (e.g., CustomEditor.py).
2. Follow the template in ExampleEditor.py:
  - Define a new edit method to implement custom logic.
  - Provide a short NAME and DESCRIPTION for the UI.
3. Register the Editor: Make sure to add your new editor to the imports in ```editors/__init__.py``` (TODO: Make this automatic). The new editor will then automatically appear in the dropdown list of editing modes.

## Installation
**Prerequisites**
- Python 3.8 or later
- CUDA-enabled GPU (optional, for acceleration)
- Git and pip

**Step-by-step Instructions**

1. Clone the Repository

```
git clone https://github.com/SvenPfiffner/SegmentBot.git
cd SegmentBot
```

2. Set Up the Environment Run the setup script provided in install.py to create a virtual environment, install dependencies, and download pre-trained models.

```
python install.py
```

3. Activate Virtual Environment On Linux/Mac:

```
source venv/bin/activate
```

3. On Windows:

```
venv\Scripts\activate
```

4. Run the Application by launching the Streamlit web app:

```
streamlit run SegmentBot/run.py
```

5. Access the app in your browser at ```http://localhost:8501```.

## Acknowledgments
[Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)

[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

[Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)

[Streamlit](https://github.com/streamlit/streamlit)
