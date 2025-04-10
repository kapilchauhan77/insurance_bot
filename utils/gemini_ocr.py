import os
import PIL.Image
from utils import config
from google import genai


os.environ['GOOGLE_CLOUD_PROJECT']=config.GCP_PROJECT_ID

os.environ['GOOGLE_CLOUD_LOCATION']=config.GCP_LOCATION

# os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'True'


def extract_text_with_gemini_flash(image_path):
    try:
        print(image_path)

        image = PIL.Image.open(image_path)
        client = genai.Client(api_key='AIzaSyAt4kDt9PHU2kdiM6glAu4_gXy2duPkoyo')
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite-001",
            contents=[
                "OCR and format in a easily readable manner",
                image
            ],
        )
        return response.text.replace('```text', '').replace('```', '').strip().replace('\n', ' ')
    except Exception as e:
        print("GEMINI Flash Error:", str(e))
        return ''


if __name__ == "__main__":
    print(extract_text_with_gemini_flash('../testocr.png'))
