from flask import Flask, request, render_template_string, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Tải mô hình
# (Trên Vercel cần đường dẫn tuyệt đối hoặc tương đối với thư mục hiện tại)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'titanic_model_pipeline.pkl')

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Giao diện HTML đơn giản
HTML_TEMPLATE = """
<!DOCTYPE htm
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Titanic Survival Predictor</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; color: #333; margin: 0; padding: 20px; }
        .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: #2c3e50; font-size: 24px; }
        .form-row { display: flex; gap: 15px; margin-bottom: 15px; }
        .form-group { flex: 1; }
        label { display: block; margin-bottom: 5px; font-weight: bold; font-size: 14px; }
        input, select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; box-sizing: border-box; font-size: 14px; }
        button { width: 100%; padding: 12px; background-color: #3498db; color: white; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; margin-top: 15px; font-weight: bold; }
        button:hover { background-color: #2980b9; }
        .result { margin-top: 20px; padding: 15px; border-radius: 5px; text-align: center; font-size: 18px; font-weight: bold; display: none; }
        .survived { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .perished { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚢 Dự Đoán Cơ Hội Sống Sót - Titanic</h1>
        <form id="predictForm">
            <div class="form-row">
                <div class="form-group">
                    <label>Hạng vé (Pclass):</label>
                    <select name="pclass"><option value="1">Hạng 1</option><option value="2">Hạng 2</option><option value="3" selected>Hạng 3</option></select>
                </div>
                <div class="form-group">
                    <label>Giới tính (Sex):</label>
                    <select name="sex"><option value="male">Nam</option><option value="female">Nữ</option></select>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label>Tuổi (Age):</label>
                    <input type="number" name="age" value="28" min="1" max="100">
                </div>
                <div class="form-group">
                    <label>Danh xưng (Title):</label>
                    <select name="title"><option value="Mr">Mr</option><option value="Miss">Miss</option><option value="Mrs">Mrs</option><option value="Master">Master</option><option value="Other">Other</option></select>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label>Giá vé (Fare, £):</label>
                    <input type="number" name="fare" value="32.0" step="0.1">
                </div>
                <div class="form-group">
                    <label>Cảng lên tàu (Embarked):</label>
                    <select name="embarked"><option value="S">Southampton</option><option value="C">Cherbourg</option><option value="Q">Queenstown</option></select>
                </div>
            </div>

            <div class="form-row">
                <div class="form-group">
                    <label>Số người thân (SibSp):</label>
                    <input type="number" name="sibsp" value="0" min="0">
                </div>
                <div class="form-group">
                    <label>Cha mẹ/con cái (Parch):</label>
                    <input type="number" name="parch" value="0" min="0">
                </div>
            </div>

            <button type="submit">🚀 Dự Đoán Ngay</button>
        </form>
        <div id="resultBox" class="result"></div>
    </div>
    
    <script>
        document.getElementById('predictForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData);
            
            const btn = e.target.querySelector('button');
            btn.innerText = 'Đang dự đoán...';
            btn.disabled = true;
            
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                
                const resBox = document.getElementById('resultBox');
                resBox.style.display = 'block';
                
                if (result.error) {
                    resBox.className = 'result perished';
                    resBox.innerText = 'Lỗi: ' + result.error;
                } else {
                    if (result.prediction === 1) {
                        resBox.className = 'result survived';
                        resBox.innerText = `🎉 CÓ KHẢ NĂNG SỐNG SÓT! (Xác suất: ${result.probability}%)`;
                    } else {
                        resBox.className = 'result perished';
                        resBox.innerText = `☠️ KHÔNG SỐNG SÓT (Xác suất: ${result.probability}%)`;
                    }
                }
            } catch (err) {
                alert('Lỗi kết nối tới server.');
            } finally {
                btn.innerText = '🚀 Dự Đoán Ngay';
                btn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Không tìm thấy file mô hình trên server.'}), 500
        
    try:
        data = request.json
        
        # Parse inputs
        age = float(data.get('age', 28))
        fare = float(data.get('fare', 32))
        pclass = int(data.get('pclass', 3))
        sex = data.get('sex', 'male')
        embarked = data.get('embarked', 'S')
        title = data.get('title', 'Mr')
        sibsp = int(data.get('sibsp', 0))
        parch = int(data.get('parch', 0))
        
        # Calculate Family_category
        family_size = sibsp + parch + 1
        if family_size == 1:
            family_cat = "Single"
        elif family_size <= 4:
            family_cat = "Small"
        elif family_size <= 6:
            family_cat = "Medium"
        else:
            family_cat = "Large"
            
        # Create DataFrame for prediction
        input_data = pd.DataFrame([{
            'Age': age,
            'Fare': fare,
            'Pclass': pclass,
            'Sex': sex,
            'Embarked': embarked,
            'Title': title,
            'Family_category': family_cat
        }])
        
        # Predict
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        survival_prob = round(probabilities[1] * 100, 1)
        
        return jsonify({
            'prediction': prediction,
            'probability': survival_prob
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Vercel đòi hỏi biến `app` (WSGI callable)
if __name__ == '__main__':
    app.run(debug=True)
