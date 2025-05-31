# Exercise_monitering_with_openpifpaf
# Exercise Form Analysis System

![OpenPifPaf Keypoints](https://example.com/path/to/sample_keypoints_image.gif) *(Replace with actual sample image)*

A hybrid system combining LSTM and rule-based methods to analyze exercise form, detect mistakes, and count repetitions for:
- Squats
- Pushups
- Situps
- Lunges
- Planks (form analysis only)

## Key Features
✔ Real-time exercise form analysis  
✔ Repetition counting for dynamic exercises  
✔ Detailed mistake identification  
✔ Hybrid LSTM + Rule-based detection  
✔ Multi-angle video processing  

Installation
# Clone repository
git clone https://github.com/yourusername/exercise-analysis.git
cd exercise-analysis

# Install dependencies
pip install -r requirements.txt

# Download pretrained models
wget https://example.com/path/to/models.zip && unzip models.zip


Usage
1.Process a video file:
python analyze.py --input sample_video.mp4 --exercise pushup
2.Available options:
--input        Path to video file
--exercise     Type of exercise (squat,pushup,situp,lunge,plank)
--output_dir   Directory to save results (default: ./results)
--show         Display real-time visualization


Output Samples
Console Output:
Processed 300 frames (10.0 FPS)
Detected 12 pushup repetitions
Form errors detected:
- Frame 45: Elbows flaring >45°
- Frame 128: Incomplete range of motion
- Frame 215: Hip sagging


Generated Files:
results/
   ├── sample_video_annotated.mp4
   ├── sample_video_keypoints.npy
   └── sample_video_report.json

Training Your Own Model
Prepare dataset of annotated videos

Preprocess keypoints:

python
python preprocess.py --data_dir ./videos --output ./training_data
Train LSTM:

python
python train.py --data ./training_data --epochs 50
Limitations
Requires clear view of full body

Best results with front/side angles

Minimum 720p resolution recommended

Contributors
[Karthikeya] 
[Lakshmi]
[Swedha]
[Sandip]
[mohasin]
[sreehari]

License
MIT License - See LICENSE.md



   
