# services/ml_interface.py
from PIL import Image
from io import BytesIO
import base64
import random

def process_image(file_obj):
    """
    1) Open with PIL
    2) 'Process' it (placeholder: just copy or resize)
    3) Create random ratio, classify as Mild/Moderate/Severe
    4) Return (base64_image_string, ratio, severity)
    """
    img = Image.open(file_obj)
    img = img.resize((200, 200))  # just resize for clarity

    # Convert image to base64 for embedding in HTML
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    processed_bytes = buffer.getvalue()
    base64_str = base64.b64encode(processed_bytes).decode('utf-8')

    # generate random ratio
    ratio = round(random.uniform(0.0, 0.15), 3)

    # classify
    if ratio < 0.01:
        severity = "None"
    elif ratio <= 0.05:
        severity = "Mild"
    elif ratio <= 0.10:
        severity = "Moderate"
    else:
        severity = "Severe"

    return base64_str, ratio, severity
