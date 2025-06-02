# Exercise Monitoring with OpenPifPaf  
## Exercise Form Analysis System

![OpenPifPaf Keypoints](https://example.com/path/to/sample_keypoints_image.gif)  
*Replace with an actual sample image or GIF*

A hybrid system that combines LSTM-based classification and rule-based analysis to monitor exercise form, detect common mistakes, and count repetitions for the following exercises:

- ✅ Squats  
- ✅ Pushups  
- ✅ Situps  
- ✅ Lunges  
- ✅ Planks (form analysis only)

---

## 🚀 Key Features

- Real-time exercise form analysis  
- Repetition counting for dynamic exercises  
- Detailed mistake identification with visual feedback  
- Hybrid detection using LSTM + rule-based logic  
- Support for multi-angle (front/side) video input  

---

## 🛠️ Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/exercise-analysis.git
cd exercise-analysis
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Download Pretrained Models
```bash
wget https://example.com/path/to/models.zip
unzip models.zip
```

---

## ▶️ Usage

### Analyze a Video File
```bash
python analyze.py --input sample_video.mp4 --exercise pushup
```

### Options
- `--input`: Path to the input video file  
- `--exercise`: Type of exercise (`squat`, `pushup`, `situp`, `lunge`, `plank`)  
- `--output_dir`: Directory to save output (default: `./results`)  
- `--show`: Display real-time visualizations (optional)

---

## 🧾 Sample Output

### Console Log
```
Processed 300 frames (10.0 FPS)
Detected 12 pushup repetitions
Form errors detected:
- Frame 45: Elbows flaring > 45°
- Frame 128: Incomplete range of motion
- Frame 215: Hip sagging
```

### Generated Files (`./results`)
```
sample_video_annotated.mp4       # Annotated video with keypoints & analysis
sample_video_keypoints.npy       # Extracted keypoints data
sample_video_report.json         # Detailed form & rep analysis report
```

---

## 📚 Train Your Own Model

### Step 1: Preprocess Keypoints
```bash
python preprocess.py --data_dir ./videos --output ./training_data
```

### Step 2: Train LSTM Model
```bash
python train.py --data ./training_data --epochs 50
```

---

## ⚠️ Limitations

- Requires full body visibility  
- Performs best with front or side view angles  
- Minimum resolution: 720p recommended  
- Inconsistent lighting may affect keypoint detection  

---

## 👥 Contributors

- Karthikeya  
- Lakshmi  
- Swedha  
- Sandip  
- Mohasin  
- Sreehari  

---

## 📄 License

This project is licensed under the [MIT License](LICENSE.md).
