import tkinter as tk
from matplotlib.mathtext import math_to_image
from io import BytesIO
from PIL import ImageTk, Image
import sys

def render_math_on_image(
    math_expr, 
    bg_image_path, 
    output_size, 
    paste_position, 
    text_height_px
):
    # Open background image and get DPI (default to 72 if not present)
    bg_img = Image.open(bg_image_path).convert("RGBA")
    dpi = bg_img.info.get("dpi", (72, 72))[0]
    bg_img = bg_img.resize(output_size, Image.LANCZOS)

    # Render math text (will have white background)
    buffer = BytesIO()
    math_to_image(
        math_expr, 
        buffer, 
        dpi=dpi, 
        format='png'
    )
    buffer.seek(0)
    math_img = Image.open(buffer).convert("RGBA")

    # Make white pixels transparent
    datas = math_img.getdata()
    newData = []
    for item in datas:
        if item[0] > 250 and item[1] > 250 and item[2] > 250:
            newData.append((255, 255, 255, 0))  # Transparent
        else:
            newData.append(item)
    math_img.putdata(newData)

    # Resize math image to desired text height in pixels
    scale = text_height_px / math_img.height
    new_width = int(math_img.width * scale)
    math_img = math_img.resize((new_width, text_height_px), Image.LANCZOS)

    # Paste math image onto background at specified position
    bg_img.paste(math_img, paste_position, math_img)

    return bg_img

class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

    def createWidgets(self):
        # Parameters
        math_expr = 'this is alpha: $\\alpha$'
        bg_image_path = sys.argv[1]  # Specify your background image path
        output_size = (400, 300)          # Output image size (width, height)
        paste_position = (100, 100)       # Where to paste the math text (x, y)
        text_height_px = 12               # Desired text height in pixels

        # Compose image
        composed_img = render_math_on_image(
            math_expr, bg_image_path, output_size, paste_position, text_height_px
        )

        # Convert to Tkinter image
        image = ImageTk.PhotoImage(composed_img)

        self.label = tk.Label(self, image=image)
        self.label.img = image
        self.label.pack(side="bottom")
        self.QUIT = tk.Button(self, text="QUIT", fg="red", command=root.destroy)
        self.QUIT.pack(side="top")

root = tk.Tk()
app = Application(master=root)
app.mainloop()