**Simon:**
![Hackatime Time](https://hackatime-badge.hackclub.com/U08HC7N4JJW/Solvy)

**Dwait:**
![Hackatime Time](https://hackatime-badge.hackclub.com/U0847KFMUSC/Solvy)

# Solvy

**Solvy** is an AI-powered app designed to automatically solve worksheets.

> ⚠️ **Note**: This project is in **early development** and currently **not functional**. Contributions and feedback are welcome!

---

## 🛠 Requirements

- Linux (requirements require modifications to work with other systems)
- Python 3.11
- An NVIDIA GPU (for local AI inference) (optional)

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
4. **Add your API keys**
   ```python
   GEMINI_API_KEY=<YOUR_GEMINI_API_KEY>
   ROBOFLOW_API_KEY=<YOUR_ROBOFLOW_API_KEY>
   ```
