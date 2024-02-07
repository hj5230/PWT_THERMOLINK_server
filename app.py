# import os
# import numpy as np
import pandas as pd
# import joblib
# from joblib import load
from joblib import load, dump
# from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
from flask import Flask, request, jsonify
# from flask_socketio import SocketIO
from flask_cors import CORS
from flask_pymongo import PyMongo

app = Flask(__name__)
CORS(app, cors_allowed_origins="*")
# socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)
app.config["MONGO_URI"] = "mongodb://localhost:27017/yourDatabaseName"
mongo = PyMongo(app)

@app.route('/')
def flask():
    return jsonify({'message': 'Not 404'})

@app.route('/login')
def login():
    USER = mongo.db.USER
    data = USER.find_one()
    return data

# @socketio.on('audio')
# def handleAudio(data):
#     print(f'Data type: {type(data)}, Data size: {len(data) if hasattr(data, "__len__") else "N/A"}')

# 命令处理函数
def adjust_temperature(temperature):
    return {"action": "adjust_temperature", "temperature": temperature}

def schedule_heater(hours):
    return {"action": "open_heater", "hours": hours}

def schedule_heater_shutdown(hours):
    return {"action": "schedule_heater_shutdown", "hours": hours}

# 解析命令
def parse_heater_commands(text):
    commands = []
    temp_pattern = r"I'd like to see the temperature cranked up to (\d+) degrees"
    schedule_pattern = r"open the heater in (\d+) hours"
    shutdown_pattern = r"shutdown the heater in (\d+) hours"

    temp_match = re.search(temp_pattern, text, re.IGNORECASE)
    if temp_match:
        commands.append(adjust_temperature(temp_match.group(1)))

    schedule_match = re.search(schedule_pattern, text, re.IGNORECASE)
    if schedule_match:
        commands.append(schedule_heater(schedule_match.group(1)))

    shutdown_match = re.search(shutdown_pattern, text, re.IGNORECASE)
    if shutdown_match:
        commands.append(schedule_heater_shutdown(shutdown_match.group(1)))

    return commands

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
def get_contact():
    return jsonify({
        'email': 'customer.service@example.com',
        'phone': '0123456789'
    })

@app.route('/grade-app', methods=['POST'])
def grade_app():
    data = request.json
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True)
