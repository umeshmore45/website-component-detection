import requests
import base64
import json
import re

def extract_and_clean_json(full_content):
    match = re.search(r'\{.*\}', full_content, re.DOTALL)
    if not match:
        return None
    schema_str = match.group(0)
    # Remove newlines inside quotes for "text" fields
    schema_str = re.sub(
        r'("text":\s*")([^"]*?)["\n]',
        lambda m: m.group(1) + m.group(2).replace('\n', ' ') + '"',
        schema_str
    )
    # Remove any remaining control characters
    schema_str = re.sub(r'[\x00-\x1F]+', ' ', schema_str)
    return schema_str


def get_schema_from_image_streaming(image_path, prompt):
    """Alternative version that handles streaming responses and returns direct JSON output"""
    try:
        with open(image_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print("Error reading image:", e)
        return

    payload = {
        "model": "llama3.2-vision",
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_base64]
            }
        ]
    }

    try:
        response = requests.post("http://localhost:11434/api/chat", json=payload)
        response.raise_for_status()
    except Exception as e:
        print("Error during request:", e)
        return

    full_content = ""
    for line in response.text.strip().split('\n'):
        if not line.strip():
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'message' in chunk and 'content' in chunk['message']:
            full_content += chunk['message']['content']
   
    print(full_content)
    schema_str = extract_and_clean_json(full_content)
    if not schema_str:
        print("No JSON schema found in response.")
        return

    try:
        schema_json = json.loads(schema_str)
        print(json.dumps(schema_json, indent=2))
    except Exception as e:
        print("Could not parse JSON schema:", e)
        print("Raw schema string:", schema_str)


def build_prompt(filename):
    if "header" in filename.lower():
        return (
            "Analyze the attached image and filename ('Header'). "
            "Return only the component structure as JSON. "
            "Include ONLY components. "
            "For each text component, include: component name, type, text and any relevant metadata. "
        )
    elif "banner" in filename.lower():
        return (
            "Analyze the attached image and filename ('Banner'). "
            "Return only the component structure as JSON for a banner, including image, text, button, and icon components. "
            "For each component, include: type, component name, text (if any), position, size, color, and metadata. "
            "Example format: "
            "{"
            "\"type\": \"banner\", "
            "\"schema\": ["
            "  {\"component\": \"Image\", \"type\": \"image\", \"position\": [x, y], \"size\": [w, h]},"
            "  {\"component\": \"Text\", \"type\": \"text\", \"text\": \"Sale Now On!\", \"position\": [x, y], \"size\": [w, h], \"color\": \"#FF0000\"},"
            "  {\"component\": \"Button\", \"type\": \"button\", \"text\": \"Shop Now\", \"position\": [x, y], \"size\": [w, h], \"color\": \"#00FF00\"},"
            "  {\"component\": \"Icon\", \"type\": \"icon\", \"name\": \"star\", \"position\": [x, y], \"size\": [w, h], \"color\": \"#FFD700\"}"
            "]"
            "}"
        )
    else:
        return (
            "Analyze the attached image and filename. "
            "Return only the component structure as JSON for the detected type. "
            "For each component, include: type, component name, text (if any), position, size, color, and metadata."
        )

# Example usage
image_path = "/Users/umesh.more/Documents/image-dec/header_footer_project/crops/Screenshot 2025-07-11 at 11.14.36 AM/Header_2.jpg"
filename = image_path.split("/")[-1]
prompt = build_prompt(filename)


get_schema_from_image_streaming(image_path, prompt)
