from image_context_recognition import ImageContextRecognition
from image_caption_generation import GeminiMemeCaptionGeneration
from image_caption_generation import LLaMa3MemeCaptionGeneration



def main(context_recognition: ImageContextRecognition, image_caption_generation_gemini: GeminiMemeCaptionGeneration, image_caption_generation_llama3: LLaMa3MemeCaptionGeneration):
    print("Welcome to my simple meme generator app. Type 'exit' to stop.")
    while True:
        url = input("Enter a URL to download its content or 'exit' to quit: ")
        if url.lower() == 'exit':
            print("Exiting the application. Goodbye!")
            exit()
        image_description = context_recognition.generate_image_description(url)
        meme_text_gemini = image_caption_generation_gemini.generate_meme_text(image_description)
        meme_text_llama3 = image_caption_generation_llama3.generate_meme_text(image_description)
        
        print(f"Gemini Meme Caption: {meme_text_gemini}")
        print(f"LLaMA 3 Meme Caption: {meme_text_llama3}")



if __name__ == "__main__":
    image_context_recognition = ImageContextRecognition()
    image_caption_generation_gemini = GeminiMemeCaptionGeneration()
    image_caption_generation_llama3 = LLaMa3MemeCaptionGeneration()

    main(image_context_recognition, image_caption_generation_gemini, image_caption_generation_llama3)





# from image_caption_generation import GeminiMemeCaptionGeneration
# from image_caption_generation import LLaMa3MemeCaptionGeneration
