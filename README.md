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
3. **Configure the Gemini API key**
   ```python
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
