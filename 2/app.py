from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import base64
import pandas as pd
import numpy as np
import cv2
import requests
import json
import re

app = Flask(__name__)

AI_API_KEY = "sk-93416056883c477ab071d371f3762616"
AI_API_URL = "https://api.deepseek.com/v1/chat/completions"

DISEASE_MAP = {
    "none": "无",
    "diabetes": "糖尿病",
    "hypertension": "高血压",
    "hyperlipidemia": "高血脂"
}

def clean_markdown(text):
    text = re.sub(r'[#*_~\[\]]', '', text)
    text = re.sub(r'-{2,}', '-', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def get_diet_suggestion(food_name, disease):
    if disease == "none":
        prompt = f"请提供关于食物'{food_name}'的营养信息和饮食建议。请直接用简洁的文字回答，不要使用markdown格式符号。"
    else:
        disease_name = DISEASE_MAP.get(disease, disease)
        prompt = f"我患有{disease_name}，请问我可以吃'{food_name}'吗？请提供详细的饮食建议，包括：1) 是否可以食用；2) 食用时的注意事项；3) 如果不适合，有什么替代食材推荐？请直接用简洁的文字回答，不要使用markdown格式符号。"
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AI_API_KEY}"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一位专业的营养师，请提供科学、准确的饮食建议。请直接用简洁的文字回答，不要使用任何markdown格式符号（如#、*、-等）。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        response = requests.post(AI_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        
        if data.get("choices") and len(data["choices"]) > 0:
            content = data["choices"][0]["message"]["content"].strip()
            return clean_markdown(content)
        else:
            return "未能获取饮食建议"
    except Exception as e:
        print(f"AI API调用失败: {e}")
        return f"获取饮食建议失败: {str(e)}"

try:
    df = pd.read_excel('e:/cv_project/2/class_names.xlsx')
    class_names = {}
    for idx, row in df.iterrows():
        class_idx = row.iloc[0]
        chinese_name = row.iloc[1]
        english_name = row.iloc[2] if pd.notna(row.iloc[2]) else ""
        class_names[class_idx] = {
            'chinese': chinese_name,
            'english': english_name
        }
    print(f"成功加载 {len(class_names)} 个食物类别")
except Exception as e:
    print(f"加载类别文件失败: {e}")
    class_names = {}

try:
    checkpoint = torch.load('e:/cv_project/2/resmodel-32-1.069.pt', map_location='cpu')
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model_state_dict = checkpoint['model']
    else:
        model_state_dict = checkpoint

    model = models.resnet50(pretrained=False)
    model.load_state_dict(model_state_dict)
    model.eval()
    print("模型加载成功!")
except Exception as e:
    print(f"模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    model = None

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def optimize_image(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return Image.open(io.BytesIO(image_bytes)).convert('RGB')

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)

        img = cv2.bilateralFilter(img, 9, 75, 75)

        _, buffer = cv2.imencode('.jpg', img)
        optimized_bytes = buffer.tobytes()
        return Image.open(io.BytesIO(optimized_bytes)).convert('RGB')
    except Exception as e:
        print(f"图片优化失败: {e}")
        return Image.open(io.BytesIO(image_bytes)).convert('RGB')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': '模型未加载'}), 500

    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)

        if data.get('optimize', False):
            image = optimize_image(image_bytes)
        else:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_prob, top_idx = torch.topk(probabilities, 5)

        results = []
        top_food_name = ""
        for i in range(5):
            idx = top_idx[i].item()
            prob = top_prob[i].item()
            if idx in class_names:
                food_info = class_names[idx]
                class_name = f"{food_info['chinese']} ({food_info['english']})"
                if i == 0:
                    top_food_name = food_info['chinese'] if food_info['chinese'] else food_info['english']
            else:
                class_name = f"未知类别_{idx}"
            results.append({
                'class': class_name,
                'chinese': class_names[idx]['chinese'] if idx in class_names else "",
                'english': class_names[idx]['english'] if idx in class_names else "",
                'probability': round(prob * 100, 2)
            })

        disease = data.get('disease', 'none')
        diet_suggestion = ""
        if top_food_name:
            diet_suggestion = get_diet_suggestion(top_food_name, disease)

        return jsonify({
            'predictions': results,
            'diet_suggestion': diet_suggestion,
            'disease': DISEASE_MAP.get(disease, disease)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/camera_predict', methods=['POST'])
def camera_predict():
    if model is None:
        return jsonify({'error': '模型未加载'}), 500

    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': '无法解码图片'}), 400

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_prob, top_idx = torch.topk(probabilities, 5)

        results = []
        for i in range(5):
            idx = top_idx[i].item()
            prob = top_prob[i].item()
            if idx in class_names:
                food_info = class_names[idx]
                class_name = f"{food_info['chinese']} ({food_info['english']})"
            else:
                class_name = f"未知类别_{idx}"
            results.append({
                'class': class_name,
                'chinese': class_names[idx]['chinese'] if idx in class_names else "",
                'english': class_names[idx]['english'] if idx in class_names else "",
                'probability': round(prob * 100, 2)
            })

        return jsonify({'predictions': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if model is None:
        return jsonify({'error': '模型未加载'}), 500

    try:
        data = request.get_json()
        images = data.get('images', [])

        if not images:
            return jsonify({'error': '没有图片数据'}), 400

        if len(images) > 20:
            return jsonify({'error': '最多支持20张图片'}), 400

        results = []
        for idx, img_data in enumerate(images):
            try:
                image_data = img_data.split(',')[1] if ',' in img_data else img_data
                image_bytes = base64.b64decode(image_data)

                if data.get('optimize', False):
                    image = optimize_image(image_bytes)
                else:
                    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

                img_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    top_prob, top_idx = torch.topk(probabilities, 3)

                predictions = []
                for i in range(3):
                    class_idx = top_idx[i].item()
                    prob = top_prob[i].item()
                    if class_idx in class_names:
                        food_info = class_names[class_idx]
                        class_name = f"{food_info['chinese']} ({food_info['english']})"
                    else:
                        class_name = f"未知类别_{class_idx}"
                    predictions.append({
                        'class': class_name,
                        'chinese': class_names[class_idx]['chinese'] if class_idx in class_names else "",
                        'english': class_names[class_idx]['english'] if class_idx in class_names else "",
                        'probability': round(prob * 100, 2)
                    })

                results.append({
                    'index': idx,
                    'success': True,
                    'predictions': predictions
                })
            except Exception as e:
                results.append({
                    'index': idx,
                    'success': False,
                    'error': str(e)
                })

        return jsonify({'results': results, 'total': len(images)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
