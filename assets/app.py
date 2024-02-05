from flask import Flask, request, jsonify
from joblib import load
import numpy as np
from flask import Flask, request, jsonify
from joblib import load, dump
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

# 加载模型
heater_on_time_prediction_model = load('heater_on_time_prediction_model.joblib')
temperature_prediction_model = load('temperature_prediction_model.joblib')
heating_time_prediction_model = load('heating_time_prediction_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # 接收请求数据

    # 预测加热器开启时间（小时数）
    heater_on_time_pred = heater_on_time_model.predict([data['features_heater_on_time']])
    heater_on_time_pred_value = heater_on_time_pred[0] if len(heater_on_time_pred) > 0 else None

    if heater_on_time_pred_value is not None:
        # 将第一个模型的预测结果插入到第二个模型的输入特征数组中的指定位置
        features_temp_modified = data['features_temp']
        features_temp_modified.insert(1, heater_on_time_pred_value)

        # 预测目标温度
        target_temp_pred = temperature_model.predict([features_temp_modified])
        target_temp_pred_value = target_temp_pred[0] if len(target_temp_pred) > 0 else None

        if target_temp_pred_value is not None:
            # 将第二个模型的预测结果插入到第三个模型的输入特征数组中的指定位置
            features_heating_time_modified = data['features_heating_time']
            features_heating_time_modified.insert(1, target_temp_pred_value)

            # 预测所需加热时间
            heating_time_pred = heating_time_model.predict([features_heating_time_modified])
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
heater_on_time_model = load('heater_on_time_prediction_model.joblib')
temperature_model = load('temperature_prediction_model.joblib')
heating_time_model = load('heating_time_prediction_model.joblib')

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
    df = pd.read_csv('simulated_heater_usage_data.csv')
    df = pd.concat([df, new_df], ignore_index=True)
    
    # 在保存之前移除中间值列
    df_to_save = df[['date', 'time', 'target_temperature', 'initial_temperature', 'heater_on', 'temperature_outside', 'humidity', 'datetime', 'last_use_time_diff', 'avg_use_time_last_week', 'heating_time']]
    df_to_save.to_csv('simulated_heater_usage_data.csv', index=False)
    
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
    dump(heater_on_time_model, 'heater_on_time_prediction_model.joblib')
    
    # 更新目标温度模型
    y_temp = df['target_temperature']
    temperature_model.fit(df[['weekday', 'hour', 'initial_temperature']], y_temp)
    dump(temperature_model, 'temperature_prediction_model.joblib')
    
    # 更新所需加热时间模型
    y_heating = df['heating_time']
    heating_time_model.fit(df[['initial_temperature', 'target_temperature', 'temperature_outside', 'humidity']], y_heating)
    dump(heating_time_model, 'heating_time_prediction_model.joblib')
    
    return jsonify({'status': 'models updated'})
if __name__ == '__main__':
    app.run(debug=True)