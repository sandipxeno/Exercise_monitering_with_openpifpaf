# 🏋️‍♂️ Exercise Monitoring with OpenPifPaf  
## Smart Exercise Form Analysis & Repetition Counting System

![OpenPifPaf Keypoints](https://example.com/path/to/sample_keypoints_image.gif)  
> *Visualizing body keypoints using OpenPifPaf — ![Screenshot (370)](https://github.com/user-attachments/assets/d38450b8-3db2-4c74-8350-4d3602cbbde3)

---
for push up - ![Screenshot (372)](https://github.com/user-attachments/assets/50b44454-68b6-4b8d-a074-b3598ac28392)
---
for pull up - ![Screenshot (375)](https://github.com/user-attachments/assets/9c9c860b-b5ab-43bb-b6f0-d31714c3f80f)
---
for squat -- ![Screenshot (378)](https://github.com/user-attachments/assets/8210ba3f-ed0c-4321-9b48-2ca176fc029a)
---
for leg raises -- ![Screenshot (381)](https://github.com/user-attachments/assets/3e36d862-b32a-4c04-870e-6336c40f341e)
---
for plank -- ![Screenshot (384)](https://github.com/user-attachments/assets/679d0287-dae7-4317-a45a-db04a4ca3b56)
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

🧠 How It Works
The system is a hybrid pipeline combining pose estimation, LSTM-based classification, and rule-based correction to monitor exercise form and count repetitions.

🔄 System Flow
css
Copy
Edit
Input Video → OpenPifPaf → Keypoint Extraction → LSTM + Rules → Mistake Detection + Rep Counting → Output (Video + Report)
🧩 Components Breakdown
1. Keypoint Detection with OpenPifPaf
Input: Raw .mp4 video of user performing an exercise.

Tool Used: OpenPifPaf

Output: 2D coordinates of body joints (nose, shoulders, elbows, knees, ankles, etc.) frame by frame.

Stored as: video_keypoints.npy

📌 These keypoints form the base data for further analysis.

2. Repetition Counting (Dynamic Exercises)
Applies to: Squats, Pushups, Situps, Lunges

Uses joint angles (e.g., knee, elbow, hip) over time.

Detects repetition cycles using:

Slope changes

Threshold-based triggers (e.g., when knee angle dips below 90°)

Outputs total reps counted.

3. Posture Mistake Detection
Uses rule-based checks based on keypoint geometry:

Example for Pushups:

Elbow flaring > 45°

Hip lower than shoulder → “Hip Sag”

Incomplete elbow extension

Outputs frame-by-frame mistake logs like:

json
Copy
Edit
{
  "frame": 215,
  "issue": "Hip sagging"
}
4. Form Classification via LSTM
A trained LSTM (Long Short-Term Memory) model:

Input: Sequence of keypoints across frames.

Output: Binary class (Correct = 1, Incorrect = 0)

Helpful in detecting subtle mistakes not caught by rules.

You can train your own LSTM using:

bash
Copy
Edit
python train.py --data ./training_data
5. Plank Special Case
Plank has no repetitions.

System only checks for:

Straight spine (shoulder–hip–ankle alignment)

Hip sag/bend

Outputs: Whether the form is "Good" or "Poor" over time.

6. Annotated Output Generation
Generates:

📹 video_annotated.mp4: Overlayed keypoints + mistake highlights

📄 video_report.json: Summary of reps & mistakes

📊 video_keypoints.npy: Raw keypoint time-series

🧠 Sample Output Flow
bash
Copy
Edit
python analyze.py --input pushup_video.mp4 --exercise pushup
Output in ./results:

pushup_video_annotated.mp4

pushup_video_report.json

pushup_video_keypoints.npy

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
