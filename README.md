# 🏋️‍♂️ Exercise Monitoring with OpenPifPaf  
## Smart Exercise Form Analysis & Repetition Counting System

![OpenPifPaf Keypoints](https://example.com/path/to/sample_keypoints_image.gif)  
> *Visualizing body keypoints using OpenPifPaf — replace with an actual sample GIF or image*

---

A hybrid AI system using **OpenPifPaf** for keypoint detection, combined with **LSTM-based classification** and **rule-based logic** to monitor and analyze your workout form. It detects **posture mistakes**, **counts repetitions**, and provides **visual + textual feedback** for five popular exercises:

✅ **Supported Exercises**:
- Squats  
- Pushups  
- Situps  
- Lunges  
- Planks *(form analysis only)*

---

## 🚀 Features

- 📹 Real-time exercise analysis from videos  
- 🔁 Accurate repetition counter  
- ❌ Detects common form mistakes (e.g., knee position, hip sag, elbow angle)  
- 🤖 Hybrid logic: Deep Learning + Rule-based corrections  
- 🎥 Multi-angle support: Side-view & Front-view  
- 📊 Detailed result reports with annotations  

---

## 🛠️ Installation & Setup

1.  Clone the Repository
git clone 

2. Install Required Packages
pip install -r requirements.txt

3. Download Pretrained Models
wget https://example.com/path/to/models.zip
unzip models.zip -d models/

How to Use
Run the Analyzer
python analyze.py --input path/to/video.mp4 --exercise pushup

Available Options
Argument	Description
--input	Path to input video file (e.g., video.mp4)
--exercise	Type of exercise (squat, pushup, situp, lunge, plank)
--output_dir	Directory to save outputs (default: ./results)
--show	Optional flag to show real-time visualization during processing

Available Options
Argument	Description
--input	Path to input video file (e.g., video.mp4)
--exercise	Type of exercise (squat, pushup, situp, lunge, plank)
--output_dir	Directory to save outputs (default: ./results)
--show	Optional flag to show real-time visualization during processing

🧾 Sample Console Output
Processed 300 frames (10.0 FPS)
Detected 12 pushup repetitions
Form errors detected:
- Frame 45: Elbows flaring > 45°
- Frame 128: Incomplete range of motion
- Frame 215: Hip sagging

📚 Training Your Own LSTM Model
Step 1: Preprocess Videos to Extract Keypoints
bash
Copy
Edit
python preprocess.py --data_dir ./videos --output ./training_data
Step 2: Train the LSTM Classifier
bash
Copy
Edit
python train.py --data ./training_data --epochs 50
⚙️ Modify train.py to customize model architecture, dropout rate, learning rate, etc.

⚠️ Known Limitations
📏 Requires full-body visibility in video frame

🎥 Best results with front or side views

📹 Recommended resolution: 720p or higher

💡 Low lighting or obstruction may reduce accuracy

📌 Plank detection only supports form analysis (no rep counting)

👨‍💻 Contributors
Made with ❤️ by:

Karthikeya

Lakshmi

Swedha

Sandip

Mohasin

Sreehari

📄 License
This project is licensed under the MIT License.
Feel free to use, modify, and distribute for academic or commercial purposes.

💬 Feedback, Support & Contributions
Found a bug? → Open an Issue

Want to contribute? → Fork the repo and submit a PR

Enjoyed the project? → ⭐ Star this repository!

markdown
Copy
Edit

---

### ✅ Next Steps for You

1. **Replace**:
   - The image/GIF link under the banner.
   - The pretrained model URL.
   - The GitHub repo username (`yourusername`) in links.
2. **Save** this as `README.md` in your project root.
3. **Push** to GitHub:
   ```bash
   git add README.md
   git commit -m "Added complete project README"
   git push origin main
