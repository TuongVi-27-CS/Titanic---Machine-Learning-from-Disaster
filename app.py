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
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dự Đoán Khả Năng Sống Sót - Titanic Survival Prediction</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        dark: {
                            main: '#060609', /* Màu nền chính cực tối giống ảnh */
                            card: '#111218', /* Màu nền của thẻ panel */
                            input: '#1A1C23' /* Màu nền ô input */
                        },
                        brand: {
                            DEFAULT: '#4F46E5', /* Màu xanh indigo chủ đạo ở nút bấm và text */
                            light: '#818CF8'
                        }
                    }
                }
            }
        }
    </script>
</head>
<body class="bg-dark-main min-h-screen text-gray-100 font-sans flex items-center justify-center p-4">

    <div class="max-w-4xl w-full bg-dark-card rounded-2xl shadow-2xl border border-white/5 overflow-hidden grid grid-cols-1 md:grid-cols-12">
        
        <!-- CỘT TRÁI: FORM NHẬP THÔNG TIN (7 phần) -->
        <div class="md:col-span-7 p-6 sm:p-8 flex flex-col justify-between">
            <div>
                <!-- Tiêu đề -->
                <div class="mb-6 border-b border-white/5 pb-4">
                    <span class="text-xs uppercase tracking-widest text-brand-light font-semibold">Machine Learning Web App</span>
                    <h1 class="text-2xl sm:text-3xl font-bold tracking-wide mt-1 text-white">Titanic Survival Predictor</h1>
                    <p class="text-sm text-gray-400 mt-1">Nhập thông tin hành khách để dự đoán khả năng sống sót.</p>
                </div>

                <!-- Form Nhập Liệu -->
                <form id="predictionForm" class="space-y-4">
                    <div class="grid grid-cols-2 gap-4">
                        <!-- Pclass -->
                        <div>
                            <label class="block text-xs font-medium text-gray-400 mb-1">Hạng Vé (Pclass)</label>
                            <select id="pclass" class="w-full bg-dark-input border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-brand">
                                <option value="1">Hạng nhất (1st Class)</option>
                                <option value="2">Hạng nhì (2nd Class)</option>
                                <option value="3" selected>Hạng ba (3rd Class)</option>
                            </select>
                        </div>
                        <!-- Sex -->
                        <div>
                            <label class="block text-xs font-medium text-gray-400 mb-1">Giới tính (Sex)</label>
                            <select id="sex" class="w-full bg-dark-input border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-brand">
                                <option value="male">Nam (Male)</option>
                                <option value="female" selected>Nữ (Female)</option>
                            </select>
                        </div>
                    </div>

                    <div class="grid grid-cols-2 gap-4">
                        <!-- Age -->
                        <div>
                            <label class="block text-xs font-medium text-gray-400 mb-1">Tuổi (Age)</label>
                            <input type="number" id="age" value="25" min="0" max="100" class="w-full bg-dark-input border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-brand">
                        </div>
                        <!-- Fare -->
                        <div>
                            <label class="block text-xs font-medium text-gray-400 mb-1">Giá vé ($)</label>
                            <input type="number" id="fare" value="32.5" step="0.1" class="w-full bg-dark-input border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-brand">
                        </div>
                    </div>

                    <div class="grid grid-cols-2 gap-4">
                        <!-- SibSp -->
                        <div>
                            <label class="block text-xs font-medium text-gray-400 mb-1">Số anh chị em/vợ chồng (SibSp)</label>
                            <input type="number" id="sibsp" value="0" min="0" class="w-full bg-dark-input border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-brand">
                        </div>
                        <!-- Parch -->
                        <div>
                            <label class="block text-xs font-medium text-gray-400 mb-1">Số cha mẹ/con cái (Parch)</label>
                            <input type="number" id="parch" value="0" min="0" class="w-full bg-dark-input border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-brand">
                        </div>
                    </div>

                    <div class="grid grid-cols-2 gap-4">
                        <!-- Embarked -->
                        <div>
                            <label class="block text-xs font-medium text-gray-400 mb-1">Cảng lên tàu (Embarked)</label>
                            <select id="embarked" class="w-full bg-dark-input border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-brand">
                                <option value="S">Southampton</option>
                                <option value="C">Cherbourg</option>
                                <option value="Q">Queenstown</option>
                            </select>
                        </div>
                        <!-- Title -->
                        <div>
                            <label class="block text-xs font-medium text-gray-400 mb-1">Danh xưng (Title)</label>
                            <select id="title" class="w-full bg-dark-input border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-brand">
                                <option value="Mr">Mr</option>
                                <option value="Miss">Miss</option>
                                <option value="Mrs">Mrs</option>
                                <option value="Master">Master</option>
                                <option value="Other">Other</option>
                            </select>
                        </div>
                    </div>
                </form>
            </div>

            <!-- Nút bấm dự đoán -->
            <div class="mt-6">
                <button type="button" onclick="predictSurvival()" class="w-full bg-brand hover:bg-brand-light text-white font-semibold py-3 px-4 rounded-xl shadow-lg transition transform active:scale-95">
                    Dự Đoán Ngay (Predict)
                </button>
            </div>
        </div>

        <!-- CỘT PHẢI: HIỂN THỊ KẾT QUẢ (5 phần) -->
        <div class="md:col-span-5 bg-black/20 p-6 sm:p-8 flex flex-col justify-between border-t md:border-t-0 md:border-l border-white/5">
            <div>
                <h2 class="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <span class="w-2 h-2 rounded-full bg-brand-light animate-ping"></span>
                    Kết Quả Dự Đoán
                </h2>

                <!-- Khung hiển thị trạng thái chính -->
                <div id="resultCard" class="bg-white/5 rounded-xl p-6 text-center border border-white/5 transition-all duration-300">
                    <div id="statusIcon" class="text-4xl mb-2">⏳</div>
                    <div id="statusText" class="text-xl font-bold text-gray-400">CHƯA CÓ KẾT QUẢ</div>
                    <p id="statusDesc" class="text-xs text-gray-500 mt-1">Vui lòng nhập thông tin và bấm dự đoán.</p>
                </div>

                <!-- Thanh xác suất -->
                <div class="mt-6 space-y-2">
                    <div class="flex justify-between text-xs text-gray-400">
                        <span>Độ tin cậy / Xác suất sống sót</span>
                        <span id="probValue" class="font-bold text-brand-light">--%</span>
                    </div>
                    <div class="w-full bg-dark-main rounded-full h-2.5 overflow-hidden border border-white/5">
                        <div id="probBar" class="bg-gradient-to-r from-brand to-blue-400 h-2.5 rounded-full transition-all duration-500" style="width: 0%"></div>
                    </div>
                </div>

                <!-- Gợi ý nhanh / Thống kê nhỏ -->
                <div class="mt-6 text-xs text-gray-500 space-y-1">
                    <p>💡 <span class="text-gray-400 font-medium">Mẹo:</span> Giới tính nữ và hạng vé cao (1st Class) có tỷ lệ sống sót vượt trội hơn trong mô hình phân tích Titanic.</p>
                </div>
            </div>

            <!-- Footer nhỏ -->
            <div class="mt-6 pt-4 border-t border-white/5 text-center text-xs text-gray-600">
                HCMUT Academic Project &bull; Dark UI Theme
            </div>
        </div>

    </div>

    <script>
        async function predictSurvival() {
            const btn = document.querySelector('button');
            const originalBtnText = btn.innerText;
            btn.innerText = 'Đang dự đoán...';
            btn.disabled = true;

            const data = {
                pclass: parseInt(document.getElementById('pclass').value),
                sex: document.getElementById('sex').value,
                age: parseFloat(document.getElementById('age').value),
                fare: parseFloat(document.getElementById('fare').value),
                sibsp: parseInt(document.getElementById('sibsp').value),
                parch: parseInt(document.getElementById('parch').value),
                embarked: document.getElementById('embarked').value,
                title: document.getElementById('title').value
            };

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                const result = await response.json();

                const resultCard = document.getElementById('resultCard');
                const statusIcon = document.getElementById('statusIcon');
                const statusText = document.getElementById('statusText');
                const statusDesc = document.getElementById('statusDesc');
                const probValue = document.getElementById('probValue');
                const probBar = document.getElementById('probBar');

                if (result.error) {
                    statusIcon.innerText = '❌';
                    statusText.innerText = 'LỖI HỆ THỐNG';
                    statusText.className = 'text-xl font-bold text-rose-500';
                    statusDesc.innerText = result.error;
                    resultCard.className = 'bg-rose-950/20 rounded-xl p-6 text-center border border-rose-500/20 transition-all duration-300';
                    probValue.innerText = '0%';
                    probBar.style.width = '0%';
                } else {
                    const probability = result.probability;
                    probValue.innerText = probability + '%';
                    probBar.style.width = probability + '%';

                    if (result.prediction === 1) {
                        statusIcon.innerText = '🎉';
                        statusText.innerText = 'CÓ THỂ SỐNG SÓT';
                        statusText.className = 'text-xl font-bold text-emerald-400';
                        statusDesc.innerText = 'Hành khách có khả năng cao vượt qua thảm họa.';
                        resultCard.className = 'bg-emerald-950/20 rounded-xl p-6 text-center border border-emerald-500/20 transition-all duration-300';
                    } else {
                        statusIcon.innerText = '⚠️';
                        statusText.innerText = 'NGUY CƠ CAO KHÔNG QUA KHỎI';
                        statusText.className = 'text-xl font-bold text-rose-500';
                        statusDesc.innerText = 'Xác suất thấp dựa trên các yếu tố rủi ro.';
                        resultCard.className = 'bg-rose-950/20 rounded-xl p-6 text-center border border-rose-500/20 transition-all duration-300';
                    }
                }
            } catch (err) {
                alert('Lỗi kết nối tới server.');
            } finally {
                btn.innerText = originalBtnText;
                btn.disabled = false;
            }
        }
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
