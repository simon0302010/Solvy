**Simon:**
![Hackatime Time](https://hackatime-badge.hackclub.com/U08HC7N4JJW/Solvy)

**Dwait:**
![Hackatime Time](https://hackatime-badge.hackclub.com/U0847KFMUSC/Solvy)

![Repo Size](https://img.shields.io/github/repo-size/simon0302010/Solvy)
![Last Commit](https://img.shields.io/github/last-commit/simon0302010/Solvy)

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
   If you are using an non-NVIDIA GPU or a CPU:
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
4. **Run the program**
   ```bash
   python app.py
   ```
