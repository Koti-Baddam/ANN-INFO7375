from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

def create_handwritten_digits(output_dir, font_path, image_size=(20, 20)):
    os.makedirs(output_dir, exist_ok=True)
    for digit in range(10):
        for variation in range(10):
            img = Image.new('L', image_size, color=255)  # Grayscale (white background)
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(font_path, 16)
            draw.text((3, 2), str(digit), font=font, fill=0)  # Write digit in black
            img = img.rotate(np.random.uniform(-15, 15))  # Slight rotation for variation
            img.save(f"{output_dir}/{digit}_{variation}.png")

def load_images(image_dir):
    """
    Loads images from the specified directory and returns the data and labels.
    :param image_dir: Directory containing the images.
    :return: Tuple of (image data as numpy array, labels as numpy array).
    """
    images = []
    labels = []
    for filename in os.listdir(image_dir):
        label = int(filename.split('_')[0])  # Extract label from filename
        img = Image.open(os.path.join(image_dir, filename)).resize((20, 20))
        images.append(np.array(img).flatten() / 255.0)  # Normalize pixel values
        labels.append(label)
    return np.array(images).T, np.array(labels)


if __name__ == "__main__":
    # Create training images
    create_handwritten_digits(output_dir="train_digits", font_path="arial.ttf")

    # Create test images (unlabeled)
    create_handwritten_digits(output_dir="test_digits", font_path="arial.ttf")
