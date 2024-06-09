import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import string
import os

class pattern_match:
    def __init__(self, font_path, image_size):
        self.font_path = font_path
        self.image_size = image_size
        self.font_size = 1.3*image_size

        self.patterns = self.generatePatterns()
        self.letters = {k: v for k, v in self.patterns.items() if k in string.ascii_uppercase}
        [cv2.imwrite(f"./licensePlateOCR/assets/patterns/{char}.jpg", img) for char, img in self.patterns.items()]

    def generatePatterns(self):
        font = ImageFont.truetype(self.font_path, self.font_size)
        characters = string.digits + string.ascii_uppercase
        characters = characters.replace("Q", "") # Remove Q, not in polish license plates

        patterns = {}
        for char in characters:
            pil_img = Image.new("L", (self.image_size, self.image_size))
            draw = ImageDraw.Draw(pil_img)
            
            # Get text size
            x1, y1, x2, y2 = draw.textbbox((0, 0), char, font=font)
            w, h = x2 - x1, y2 - y1
            # center text
            x, y = (pil_img.width - w) // 2, (pil_img.height - h) //2
            draw.text((x, y), char, fill=(255), font=font)
            
            opencv_img = np.array(pil_img, dtype=np.uint8)
            inverted = cv2.bitwise_not(opencv_img) # invert image, to get black text

            patterns[char] = inverted
        return patterns
    
    def match(self, image, only_letters=False):
        detected = []
        patterns = self.letters if only_letters else self.patterns
        for char, pattern in patterns.items():
            image_scaled = cv2.resize(image, pattern.shape)
            res = cv2.matchTemplate(image_scaled, pattern,cv2.TM_CCOEFF) 
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            detected.append((char, max_val))
        detected = sorted(detected, key=lambda a: a[1], reverse=True)
        return detected[0][0]
