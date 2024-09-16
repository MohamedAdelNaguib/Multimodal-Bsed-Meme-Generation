
from modules.image_context_recognition import ImageContextRecognition
from modules.image_caption_generation import GeminiMemeCaptionGeneration
from modules.image_caption_generation import LLaMa3MemeCaptionGeneration
from modules.image_meme_assembler import MemeAssembler        

        
from flask import Flask, request, render_template
from PIL import Image
from io import BytesIO
import base64
app = Flask(__name__)

def make_meme(image: Image.Image, caption: str):
    meme = MemeAssembler(caption, image)
    img = meme.draw()
    return img

image_context_recognition = ImageContextRecognition()
image_caption_generation_gemini = GeminiMemeCaptionGeneration()
image_caption_generation_llama3 = LLaMa3MemeCaptionGeneration()


@app.route('/', methods=['GET', 'POST'])
def home():
    error_message = None
    image_data = None
    if request.method == 'POST':
        image_url = request.form['image_url']
        model = request.form['model']

        try:
            image, image_description = image_context_recognition.generate_description(image_url)
            print(image_description)
            if model == 'Gemini-Flash':
                meme_text = image_caption_generation_gemini.generate_meme_text(image_description)["content"]
            elif model == 'LLaMa-3':
                meme_text = image_caption_generation_llama3.generate_meme_text(image_description)["caption"]
            meme_image = make_meme(image, meme_text)
            buffer = BytesIO()
            meme_image.save(buffer, format="PNG")
            buffer.seek(0)
            image_data = base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            error_message = str(e)
    return render_template('index.html', image_data=image_data, error_message=error_message)

        

if __name__ == '__main__':
    app.run(debug=True)