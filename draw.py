from tkinter import *
from PIL import Image, ImageDraw
from torchvision import transforms
import numpy as np
import torch

class Drawable:
    def __init__(self, width, height, zoom, color='black', background='white', brush_size=3):
        self.width = width
        self.height = height
        self.color = color
        self.background = background
        self.brush_size = brush_size
        self.zoom = zoom
        self.points = []

        # Initialize main window
        self.root = Tk()
        self.root.title("Draw on Image")

        # Create a canvas to draw on
        self.canvas = Canvas(self.root, width=self.width * self.zoom, height=self.height * self.zoom, bg="white")
        self.canvas.pack()

        # Bind mouse drag event to canvas
        self.canvas.bind("<B1-Motion>", self.record_point)

        # Bind mouse release event to canvas
        self.canvas.bind("<ButtonRelease-1>", self.clear_points)

    def record_point(self, event):
        self.points.append((event.x, event.y))
        self.draw(event)

    def draw(self, event):
        x1, y1 = (event.x - self.brush_size), (event.y - self.brush_size)
        x2, y2 = (event.x + self.brush_size), (event.y + self.brush_size)
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.color, outline=self.color)

    def clear_points(self, event):
        self.canvas.update()
        self.root.update()

    def save(self, path):
        # Create an empty image
        image = Image.new("RGB", (self.width * self.zoom, self.height * self.zoom), self.background)
        draw = ImageDraw.Draw(image)

        # Draw the recorded points onto the image
        for point in self.points:
            draw.rectangle([point[0] - self.brush_size, point[1] - self.brush_size,
                            point[0] + self.brush_size, point[1] + self.brush_size],
                           fill=self.color, outline=self.color)

        # Save the image
        print("Saving the image...")
        image.save(path)
        print("Image saved.")

    def create_tensor(self, w, h):
        # Create an empty image
        image = Image.new("RGB", (self.width * self.zoom, self.height * self.zoom), self.background)
        draw = ImageDraw.Draw(image)

        # Draw the recorded points onto the image
        for point in self.points:
            draw.rectangle([point[0] - self.brush_size, point[1] - self.brush_size,
                            point[0] + self.brush_size, point[1] + self.brush_size],
                        fill=self.color, outline=self.color)

        # Resize the image to the desired size
        image = image.resize((w, h))

        # Convert the image to grayscale
        image = image.convert('L')

        # Convert the image to a numpy array
        image_array = np.array(image)

        # Invert the colors to represent black marks on white background
        image_array = 255 - image_array

        # Normalize the image from -1 to 1
        image_array = (image_array / 255.0 - 0.5) * 2.0

        # Convert the numpy array to a PyTorch tensor
        image_tensor = torch.tensor(image_array, dtype=torch.float)

        # Reshape the tensor to match the expected input shape (batch_size, input_size)
        image_tensor = image_tensor.view(1, w * h)

        return image_tensor



    def start(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()
        print('Window closed')

    def on_close(self):
        self.root.destroy()
        self.root.quit()

