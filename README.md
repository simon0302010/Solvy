![Repo Size](https://img.shields.io/github/repo-size/simon0302010/Solvy)
![Last Commit](https://img.shields.io/github/last-commit/simon0302010/Solvy)


**Simon:**
![Hackatime Time](https://hackatime-badge.hackclub.com/U08HC7N4JJW/Solvy)

**Dwait:**
![Hackatime Time](https://hackatime-badge.hackclub.com/U0847KFMUSC/Solvy)


<div align="left">
  <a href="https://shipwrecked.hackclub.com/?t=ghrm" target="_blank">
    <img src="https://hc-cdn.hel1.your-objectstorage.com/s/v3/739361f1d440b17fc9e2f74e49fc185d86cbec14_badge.png" 
         alt="This project is part of Shipwrecked, the world's first hackathon on an island!" 
         style="width: 35%;">
  </a>
</div>


# Solvy

**Solvy** is an AI-powered app designed to automatically solve worksheets.

> ⚠️ **Note**: Running the Machine Learning Model locally is currently broken.

---

## 🛠 Requirements

- Linux (requirements require modifications to work with other systems)
- Python 3.11

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/simon0302010/Solvy.git
   cd Solvy
   ```
2. **Install dependencies**
   
   If you are using an NVIDIA GPU:
   ```bash
   pip install -r requirements.txt
   ```
   If you are using a non-NVIDIA GPU or a CPU:
   ```bash
   pip install -r requirements_cpu.txt
   ```
   If you are not running the inference locally:
   ```bash
   pip install -r requirements_cloud.txt
   ```
3. **Add your API keys**
   ```python
   GEMINI_API_KEY=<YOUR_GEMINI_API_KEY>
   ROBOFLOW_API_KEY=<YOUR_ROBOFLOW_API_KEY>
   GEMINI_API_KEY_LIST=[<GEMINI_API_KEY_1>, <GEMINI_API_KEY_2>] (optional)
   ```

4. **Configure Solvy**

   Open `backend.py` and set the following variables at the top of the file:

   ```python
   # Inference location: "roboflow" (cloud, recommended) or "local" (experimental)
   inference = "roboflow"

   # Save processed images and results (True/False)
   save_images = True

   # OCR method: "bounding_boxes" (fastest), "tesseract" (medium), "easyocr" (slowest, but most accurate),
   ocr = "bounding_boxes"
   ```

5. **Run the program**
   ```bash
   python app.py
   ```
