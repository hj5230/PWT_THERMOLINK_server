import os
from datetime import timedelta
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import pandas as pd
from joblib import load, dump
from sklearn.preprocessing import StandardScaler

from models.User import User

load_dotenv()

app = Flask(os.getenv('FLASK_APP'))
CORS(app, cors_allowed_origins="*")
# socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)
app.config["JWT_SECRET_KEY"] = os.getenv('JWT_SECRET_KEY')
app.config["MONGO_URI"] = os.getenv('MONGO_URI')

mongo = PyMongo(app)
jwt = JWTManager(app)

@app.route('/')
def flask():
    return jsonify({'message': 'Not 404'})

@app.route('/sign', methods=['POST'])
def sign():
    # print(request)
    # if not request.is_json:
    #     return jsonify({"msg": "Missing JSON in request"}), 400
    data = request.get_json()
    username = data.get('username', None)
    email = data.get('email', None)
    product_id = data.get('productId', None)
    if not username or not email or not product_id:
        return jsonify({"msg": "Missing username, email or productId"}), 400
    user = mongo.db.users.find_one({"username": username})
    if not user:
        mongo.db.users.insert_one({
            "username": username,
            "email": email,
            "productId": product_id
        })
    jwt = create_access_token(identity=username, expires_delta=timedelta(days=7))
    return jsonify(jwt=jwt), 200

# @socketio.on('audio')
# def handleAudio(data):
#     print(f'Data type: {type(data)}, Data size: {len(data) if hasattr(data, "__len__") else "N/A"}')


# 定义命令处理函数
def adjust_temperature(temperature):
    return {"action": "adjust_temperature", "temperature": temperature}

def schedule_heater(hours):
    return {"action": "open_heater", "hours": hours}

def schedule_heater_shutdown(hours):
    return {"action": "schedule_heater_shutdown", "hours": hours}

# 解析命令，包含多种表达方式
def parse_heater_commands(text):
    commands = []
    # 扩展多种表达方式的模式（英文和中文）
    temp_patterns = [
        r"temperature cranked up to (\d+) degrees",
        r"increase temperature to (\d+) degrees",
        r"set the temperature to (\d+)",
        r"将温度调整到(\d+)度",
        r"温度提高到(\d+)度"
    ]
    schedule_patterns = [
        r"open the heater in (\d+) hours",
        r"activate heater after (\d+) hours",
        r"turn on the heater in (\d+) hours",
        r"(\d+)小时后开启加热器",
        r"在(\d+)小时后启动加热器"
    ]
    shutdown_patterns = [
        r"shutdown the heater in (\d+) hours",
        r"turn off the heater after (\d+) hours",
        r"deactivate heater in (\d+) hours",
        r"schedule the heater to shutdown in (\d+) hours",
        r"(\d+)小时后关闭加热器",
        r"在(\d+)小时后关闭加热设备"
    ]

    # 合并英文和中文模式
    all_patterns = [
        (temp_patterns, adjust_temperature),
        (schedule_patterns, schedule_heater),
        (shutdown_patterns, schedule_heater_shutdown),
    ]

    # 遍历模式并匹配命令
    for patterns, action in all_patterns:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                commands.append(action(*match.groups()))
                break  # 匹配到一个模式后即跳出循环，防止重复添加命令

    return commands

# 示例文本和对应的解析结果
example_texts = [
    "I'd like the temperature cranked up to 25 degrees.",
    "请将温度调整到25度。",
    "Please turn on the heater in 3 hours.",
    "3小时后开启加热器。",
    "Schedule the heater to shutdown in 5 hours.",
    "请在5小时后关闭加热器。"
]

# 用 AssemblyAI 进行语音转文字转录
def transcribe_audio(audio_path):
    aai.settings.api_key = "your_assemblyai_api_key"
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path)
    if transcript.status == 'completed':
        return transcript.text
    else:
        print("Transcription is not completed. Status:", transcript.status)
        return None

# 组合上传和处理逻辑
@app.route('/audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    file_path = os.path.join('uploads', audio_file.filename)
    audio_file.save(file_path)

    # 进行音频转录
    transcript_text = transcribe_audio(file_path)
    if transcript_text:
        commands = parse_heater_commands(transcript_text)
        return jsonify({'msg': 'Audio file received and processed successfully!', 'commands': commands})
    else:
        return jsonify({'error': 'Failed to transcribe audio'}), 500

# 加载模型
heater_on_time_prediction_model = load('./assets/heater_on_time_prediction_model.joblib')
temperature_prediction_model = load('./assets/temperature_prediction_model.joblib')
heating_time_prediction_model = load('./assets/heating_time_prediction_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # 接收请求数据

    try:
        # 预测加热器开启时间（小时数）
        heater_on_time_pred = heater_on_time_prediction_model.predict([data['features_heater_on_time']])
        heater_on_time_pred_value = heater_on_time_pred[0] if len(heater_on_time_pred) > 0 else None

        # 准备第二个模型的输入
        features_temp = data['features_temp']
        if heater_on_time_pred_value is not None:
            features_temp_modified = features_temp.copy()
            features_temp_modified.insert(1, heater_on_time_pred_value)

            # 预测目标温度
            target_temp_pred = temperature_prediction_model.predict([features_temp_modified])
            target_temp_pred_value = target_temp_pred[0] if len(target_temp_pred) > 0 else None

            # 准备第三个模型的输入
            features_heating_time = data['features_heating_time']
            if target_temp_pred_value is not None:
                features_heating_time_modified = features_heating_time.copy()
                features_heating_time_modified.insert(1, target_temp_pred_value)

                # 预测所需加热时间
                heating_time_pred = heating_time_prediction_model.predict([features_heating_time_modified])
                heating_time_pred_value = heating_time_pred[0] if len(heating_time_pred) > 0 else None
            else:
                heating_time_pred_value = None
        else:
            target_temp_pred_value = None
            heating_time_pred_value = None

        # 返回预测结果
        return jsonify({
            'heater_on_time_prediction': float(heater_on_time_pred_value) if heater_on_time_pred_value is not None else 'Error: No prediction',
            'target_temperature_prediction': float(target_temp_pred_value) if target_temp_pred_value is not None else 'Error: No prediction',
            'heating_time_prediction': float(heating_time_pred_value) if heating_time_pred_value is not None else 'Error: No prediction'
        })

    except Exception as e:
        # 如果有错误，返回错误信息
        return jsonify({'error': str(e)}), 400

def preprocess_features(df):
    # 执行数据预处理步骤
    df['date'] = pd.to_datetime(df['date'])
    df['weekday'] = df['date'].dt.weekday
    df['hour'] = df['time'].apply(lambda x: int(x.split(':')[0]))
    df['minutes_of_day'] = df['time'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
    df['heater_on'] = df['heater_on'].astype(int)
    # 保持原始CSV的列
    return df[['date', 'time', 'target_temperature', 'initial_temperature', 'heater_on', 'temperature_outside', 'humidity', 'datetime', 'last_use_time_diff', 'avg_use_time_last_week', 'heating_time']]

@app.route('/update-model', methods=['POST'])
def update_model():
    new_data = request.json  # 接收前端传回的数据
    
    # 预处理新数据，但不包括新生成的列
    new_df = preprocess_features(pd.DataFrame([new_data]))
    
    # 读取CSV文件，并将新数据附加到其中
    df = pd.read_csv('./assets/simulated_heater_usage_data.csv')
    df = pd.concat([df, new_df], ignore_index=True)
    
    # 在保存之前移除中间值列
    df_to_save = df[['date', 'time', 'target_temperature', 'initial_temperature', 'heater_on', 'temperature_outside', 'humidity', 'datetime', 'last_use_time_diff', 'avg_use_time_last_week', 'heating_time']]
    df_to_save.to_csv('./assets/simulated_heater_usage_data.csv', index=False)
    
    # 现在进行预处理和训练步骤...
    # 确保在此步骤中使用的是完整的df，包括新生成的预处理列
    df_preprocessed = preprocess_features(df)
    
    # 重新训练所有模型
    # 注意：确保你使用正确的特征列来训练每个模型
    def check_for_nan(df, model_name):
        if df.isnull().any().any():
            print(f"NaN values detected before training {model_name}")
        else:
            print(f"No NaN values detected before training {model_name}")
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[['weekday', 'hour', 'target_temperature', 'initial_temperature']])
    check_for_nan(df[['weekday', 'hour', 'target_temperature', 'initial_temperature']], 'heater_on_time_model')
    check_for_nan(df[['weekday', 'hour', 'initial_temperature']], 'temperature_model')
    check_for_nan(df[['initial_temperature', 'target_temperature', 'temperature_outside', 'humidity']], 'heating_time_model')
    # 更新加热器开启时间模型
    y_time = df['minutes_of_day']
    heater_on_time_model.fit(X_scaled, y_time)
    dump(heater_on_time_model, './assets/heater_on_time_prediction_model.joblib')
    
    # 更新目标温度模型
    y_temp = df['target_temperature']
    temperature_model.fit(df[['weekday', 'hour', 'initial_temperature']], y_temp)
    dump(temperature_model, './assets/temperature_prediction_model.joblib')
    
    # 更新所需加热时间模型
    y_heating = df['heating_time']
    heating_time_model.fit(df[['initial_temperature', 'target_temperature', 'temperature_outside', 'humidity']], y_heating)
    dump(heating_time_model, './assets/heating_time_prediction_model.joblib')
    
    return jsonify({'status': 'models updated'})

@app.route('/cutomer-service')
@jwt_required()
def get_contact():
    return jsonify({
        'email': 'customer.service@example.com',
        'phone': '0123456789'
    })

@app.route('/grade-app', methods=['POST'])
@jwt_required()
def grade_app():
    data = request.json # process with third-party api
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True)
