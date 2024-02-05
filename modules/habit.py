import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import datetime

def train_model(file_path):
    # Load data from Excel file
    data = pd.read_excel(file_path)

    # Features and labels
    X = data[['Current Temperature (°C)', 'External Temperature (°C)']]
    y = data[['Target Temperature (°C)', 'Estimated Heating Time (min)']]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create and train a multi-output regression model
    multioutput_model = MultiOutputRegressor(LinearRegression()).fit(X_train, y_train)
    return multioutput_model

def predict_values(model, current_temp, external_temp):
    # Predict target temperature and estimated heating time using the trained model
    predicted_values = model.predict([[current_temp, external_temp]])
    return predicted_values

def recommend_start_time(predicted_values):
    # Start Time Recommendation Logic
    peak_usage_time = datetime.datetime.strptime('07:00 AM', '%I:%M %p')
    now = datetime.datetime.now()
    recommended_start_datetime = peak_usage_time - datetime.timedelta(minutes=predicted_values[0][1])
    
    # Ensure the recommended start time is not in the past
    if recommended_start_datetime < now:
        recommended_start_datetime = now + datetime.timedelta(minutes=15)
        print("Note: Recommended start time adjusted to near future as the calculation resulted in a past time.")
    
    return recommended_start_datetime

def main():
    file_path = 'D:/作业/论文/ppt/heater_simulation_data_en.xlsx'
    current_temp = 23
    external_temp = 15

    # Training model
    model = train_model(file_path)

    # Predicting values
    predicted_values = predict_values(model, current_temp, external_temp)
    print(f"Predicted Target Temperature: {predicted_values[0][0]:.2f}°C")
    print(f"Predicted Estimated Heating Time: {predicted_values[0][1]:.2f} minutes")

    # Recommending start time
    start_time = recommend_start_time(predicted_values)
    print(f"Recommended Start Time: {start_time.strftime('%I:%M %p')} on {start_time.strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()
