from PIL import Image, ImageFont
import matplotlib.pyplot as plt
import io

text = 'This is a square root: $\\sqrt{9}$ and a fraction: $\\frac{1}{2}$'

# Create matplotlib figure
fig = plt.figure()
fig.patch.set_alpha(0.0)
plt.text(0.1, 0.5, text, fontsize=30)
plt.axis('off')

# Save to buffer
buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
plt.close(fig)

# Load with Pillow and show
buf.seek(0)
image = Image.open(buf)
image.show()
