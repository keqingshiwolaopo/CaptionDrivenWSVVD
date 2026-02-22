import os
import re
import json
import requests


def get_access_token():
    """从百度API获取access_token"""
    API_Key = ''
    Secret_Key = ''
    url = f""

    response = requests.post(url)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        raise Exception("Failed to fetch access token")


def parse_video_label(filename, dataset_type):
    if dataset_type.lower() == 'xd':
        match = re.search(r'label_([A-Z0-9\-]+)\.txt$', filename)
        if not match:
            return []
        label_str = match.group(1)
        label_dict = {
            "B1": "fighting",
            "B2": "shooting",
            "B4": "riot",
            "B5": "abuse",
            "B6": "car accident",
            "G": "explosion",
            "A": "nonviolence"
        }
        return [label_dict[label] for label in label_str.split('-') if label in label_dict]

    elif dataset_type.lower() == 'ucf':
        label = filename[:-12] if len(filename) >= 12 else filename
        return [label] if label else []

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}, only xd/ucf are supported")

def is_nonviolent(filename, dataset_type):
    if dataset_type.lower() == 'xd':
        match = re.search(r'label_([A-Z0-9\-]+)\.txt$', filename)
        return match and 'A' in match.group(1)

    elif dataset_type.lower() == 'ucf':
        return 'Normal' in filename

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}, only xd/ucf are supported")


def generate_prompt(input_text, is_nonviolent, ce_type, labels):
    label_str = ', '.join(labels) if labels else 'None'

    if ce_type == 'CE1':
        if is_nonviolent:
            return (
                f"Here are three generated text descriptions of a video. Please extract and integrate the key information from the descriptions into a concise and coherent paragraph."
                f"Make sure it's under 60 words. "
                f"Please output the revised text directly. "
                f"Text: {input_text}"
            )
        else:
            return (
                f"Here are three generated text descriptions of a video. Please extract and integrate the key event information from the descriptions into a concise and coherent paragraph."
                f"Remove all descriptions of setting and atmosphere. Make sure it's under 60 words. "
                f"Please output the revised text directly. "
                f"Text: {input_text}"
            )

    elif ce_type == 'CE2':
        if is_nonviolent:
            return (
                f"This is a normal non-violent video description. Check for violent content: if any, revise it naturally to fit the video tags and remove violence; if none, keep the original."
                f"Make sure it's under 60 words. "
                f"Please output the revised text directly. "
                f"Text: {input_text}"
            )
        else:
            return (
                f" This is a violent video description. Check if violent content matches the video tags: revise only mismatched parts to fit the tags (no overstatement); if matched, keep the original."
                f"Video Tags: {label_str}. "
                f"Remove all descriptions of setting and atmosphere. Make sure it's under 60 words. "
                f"Please output the revised text directly. "
                f"Text: {input_text}"
            )

    else:
        raise ValueError(f"Unsupported enhancement type: {ce_type}, only CE1/CE2 are supported")


def process_text_files(input_dir, output_dir, dataset_type='xd', ce_type='CE2'):
    if ce_type not in ['CE1', 'CE2']:
        raise ValueError("ce_type must be 'CE1' or 'CE2'")
    if dataset_type.lower() not in ['xd', 'ucf']:
        raise ValueError("dataset_type only supports 'xd' or 'ucf'")

    os.makedirs(output_dir, exist_ok=True)
    access_token = get_access_token()
    api_url = (
        f""
        f"?access_token={access_token}"
    )

    for filename in os.listdir(input_dir):
        if not filename.endswith('.txt'):
            continue

        file_path = os.path.join(input_dir, filename)
        try:
            labels = parse_video_label(filename, dataset_type)
            print(labels)
            non_violent_flag = is_nonviolent(filename, dataset_type)
            print(f"Processing file: {filename} | Type: {'Non-Violent' if non_violent_flag else 'Violent'} | Labels: {labels} | CE: {ce_type}")

            with open(file_path, 'r', encoding='utf-8') as f:
                input_text = f.read().strip()
            if not input_text:
                print(f"Warning: {filename} is empty, skip processing")
                continue

            prompt = generate_prompt(input_text, non_violent_flag, ce_type, labels)

            payload = json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": 0.8,
                "top_p": 0.8,
                "penalty_score": 1,
                "max_output_tokens": 170,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.0
            })
            headers = {'Content-Type': 'application/json'}
            response = requests.post(api_url, headers=headers, data=payload, timeout=30)
            response.raise_for_status()
            response_data = response.json()

            if 'result' in response_data:
                result = response_data['result'].strip()
            else:
                result = f"API response error: no 'result' field | Response content: {response_data}"

            output_path = os.path.join(output_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)

        except Exception as e:
            print(f"Failed to process {filename}: {str(e)}")
            with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                f.write(f"Processing failed: {str(e)}")


if __name__ == "__main__":
    CONFIG = {
        "input_dir": "",
        "output_dir": "",
        "dataset_type": "xd",
        "ce_type": "CE2"
    }

    process_text_files(
        input_dir=CONFIG["input_dir"],
        output_dir=CONFIG["output_dir"],
        dataset_type=CONFIG["dataset_type"],
        ce_type=CONFIG["ce_type"]
    )
    print("\nAll files processed")