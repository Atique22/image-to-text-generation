import pytesseract
from PIL import Image
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the GPT-2 model and tokenizer
model_name = "gpt2"  # You can choose other GPT-2 variants like "gpt2-medium", "gpt2-large", etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Function to extract text from the image using Tesseract OCR
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text.strip()

# Function to generate text from the extracted text using GPT-2
def generate_text_from_extracted_text(extracted_text):
    input_ids = tokenizer.encode(extracted_text, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text.strip()

if __name__ == "__main__":
    # Replace 'your_image_path.jpg' with the path to your image file
    image_path = "image.png"

    # Step 1: Extract text from the image using Tesseract OCR
    extracted_text = extract_text_from_image(image_path)

    # Step 2: Generate text from the extracted text using GPT-2
    generated_text = generate_text_from_extracted_text(extracted_text)

    print("Extracted Text:", extracted_text)
    print("Generated Text:", generated_text)
