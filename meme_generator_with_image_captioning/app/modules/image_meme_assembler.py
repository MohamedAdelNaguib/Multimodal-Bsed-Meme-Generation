from PIL import Image, ImageDraw, ImageFont, ImageStat
import textwrap

class MemeAssembler:
    basewidth = 1200  # Width to make the meme
    fontBase = 50  # Base font size
    letSpacing = 9  # Space between letters
    lineSpacing = 10  # Space between lines
    stroke_width = 20  # Stroke width
    fontfile = './static/liberation_mono/LiberationMono-Bold.ttf'
    shadow_offset = (3, 3)  # Shadow offset

    def __init__(self, caption, image):
        if not isinstance(caption, str):
            raise ValueError("Caption must be a string")
        self.img = self.createImage(image)
        self.d = ImageDraw.Draw(self.img)
        self.splitCaption = textwrap.wrap(caption, width=30)  # Adjust the wrap to suit image width
        self.splitCaption.reverse()  # Draw text from bottom up

        self.fill, self.shadow_fill = self.get_text_colors()  # Dynamic text and shadow colors based on image background

        lines = len(self.splitCaption)
        fontSize = self.fontBase - (lines * 2)  # Decrease font size with more lines
        fontSize = max(fontSize, 20)  # Ensure a minimum font size
        self.fontBase = fontSize

        self.font = ImageFont.truetype(self.fontfile, size=self.fontBase)

    def get_text_colors(self):
        # Sample the average color of the image
        stat = ImageStat.Stat(self.img)
        mean = stat.mean[:3]  # Get average of RGB channels
        brightness = sum([x*x for x in mean]) / len(mean)  # Calculate brightness based on Euclidean distance

        # Choose text color for high contrast
        if brightness > 13000:  # Adjust this threshold based on your need
            return ('#000000', '#ffffff')  # Dark text, light shadow
        else:
            return ('#ffffff', '#000000')  # Light text, dark shadow
    
    def createImage(self, image):
        wpercent = self.basewidth / float(image.size[0])
        hsize = int((float(image.size[1]) * wpercent))
        return image.resize((self.basewidth, hsize), Image.Resampling.LANCZOS)

    def draw(self):
        iw, ih = self.img.size
        initial_y = ih * 0.9  # Start drawing from the bottom 10% of the image and go upwards

        for cap in self.splitCaption:
            bbox = self.d.textbbox((0, 0), cap, font=self.font)
            th = bbox[3] - bbox[1]  # Text height
            tw = bbox[2] - bbox[0]  # Text width
            x = (iw-tw) / 2  # Center the text horizontally
            y = initial_y - th / 2  # Calculate the vertical position for this line of text

            self.drawLineWithShadow(x, y, cap)
            initial_y -= (th + self.lineSpacing)  # Move up for the next line
        return self.img

    def drawLineWithShadow(self, x, y, caption):
        # Draw shadow first
        for char in caption:
            char_bbox = self.d.textbbox((x, y), char, font=self.font)
            w = char_bbox[2] - char_bbox[0]
            shadow_pos = (x + self.shadow_offset[0], y + self.shadow_offset[1])
            self.d.text(shadow_pos, char, font=self.font, fill=self.shadow_fill)
            self.d.text((x, y), char, font=self.font, fill=self.fill)
            x += w   # Adjust for letter spacing