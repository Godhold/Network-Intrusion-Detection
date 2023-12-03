import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler
from tensorflow.keras.models import load_model
import joblib
import socket
from io import StringIO

# Load the saved model
model = load_model("/Users/godholdalomenu/Desktop/NetworkIntrusionDetection/model3.h5")

# Load the saved scaler
scaler = joblib.load("/Users/godholdalomenu/Desktop/NetworkIntrusionDetection/scaler.pkl")

# Function to preprocess input data
def preprocess_data(data):
    object_columns = data.select_dtypes(include='object').columns
    label_encoder = LabelEncoder()
    for column in object_columns:
        data[column] = label_encoder.fit_transform(data[column])

    # Use the saved scaler for feature scaling
    data_scaled = scaler.transform(data)

   
    data_reshaped = data_scaled.reshape((data_scaled.shape[0], 1, data_scaled.shape[1]))

    return data_reshaped

# Streamlit App
st.title("Network Intrusion Detection App")

# Option to choose data source
use_network_source = st.checkbox("Use Network Source", value=False)

if use_network_source:
    st.warning("Note: Using network source for demonstration. Implement proper handling for your network data.")
    
    # Function to start a simple socket server
    def start_socket_server():
        st.write("Configure Network Identifiers:")
        
        # Allow the user to input network identifiers
        host = st.text_input("Enter Server Host (e.g., localhost):", "localhost")
        port = st.number_input("Enter Server Port (e.g., 12345):", value=17890, step=1)

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, port))
        server_socket.listen(1)
        st.write(f"Socket server is listening on {host}:{port}...")
        
        while True:
            client_socket, client_address = server_socket.accept()
            st.write(f"Connection from {client_address}")
            
            data = client_socket.recv(1024).decode("utf-8")
            st.subheader("Received Data:")
            st.text(data)
            
            # Convert received data to DataFrame
            received_data = pd.read_csv(StringIO(data))
            
            # Preprocess the input data
            preprocessed_data = preprocess_data(received_data)
            
            # Make predictions using the loaded model
            predictions_prob = model.predict(preprocessed_data)
            predictions_binary = (predictions_prob > 0.5).astype(int).flatten()
            
            # Display the predictions
            st.subheader("Predictions:")
            st.write(predictions_binary)
            
            # Visualize the distribution of predictions
            st.subheader("Distribution of Predictions:")
            st.bar_chart(pd.Series(predictions_binary).value_counts())
            
            client_socket.close()

    # Start the socket server when the app is run
    start_socket_server()
else:
    # Get input data from the user
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)

        # Display a sample of the uploaded data
        st.subheader("Sample of the Uploaded Data:")
        st.write(input_data.head())

        # Preprocess the input data
        preprocessed_data = preprocess_data(input_data)

        # Make predictions using the loaded model
        predictions_prob = model.predict(preprocessed_data)
        predictions_binary = (predictions_prob > 0.5).astype(int).flatten()

        
        # Map predictions to labels
        predictions_labels = ["Normal" if pred == 0 else "Anomalous" for pred in predictions_binary]



        # Visualize the distribution of predictions
        st.subheader("Distribution of Predictions:")
        st.bar_chart(pd.Series(predictions_labels).value_counts())


        # Provide additional insights or actions based on the predictions
        st.subheader("Additional Insights:")

        # Display overall prediction statistics
        prediction_stats = pd.Series(predictions_binary).value_counts()
        st.write("Overall Prediction Statistics:")
        st.write(f"Number of Normal Instances: {prediction_stats.get(0, 0)}")
        st.write(f"Number of Anomalous Instances: {prediction_stats.get(1, 0)}")

        # Display specific insights or actions based on the predictions
        if prediction_stats.get(1, 0) > 0:
            st.write("Warning: Anomalous Activity Detected!")
            # You can provide specific recommendations, actions, or alerts based on the detected anomalies
            st.write("Recommendations:")
            st.write("- Investigate the source of anomalous activity.")
            st.write("- Take appropriate security measures.")
        else:
            st.write("No Anomalous Activity Detected.")
