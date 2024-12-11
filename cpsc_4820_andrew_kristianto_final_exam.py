
import streamlit as st
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess, decode_predictions as resnet_decode
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess, decode_predictions as vgg_decode
from google.cloud import storage
import numpy as np
import matplotlib.pyplot as plt
import io
import uuid
import requests

def upload_blob(bucket_name, destination_blob_name, file):
    """
    This function is to upload the file to Google Cloud Storage bucket.
    """
    try:
        # First we initialize the storage client
        storage_client = storage.Client()

        # To get the bucket
        bucket = storage_client.bucket(bucket_name)

        # Creating blob in the bucket
        blob = bucket.blob(destination_blob_name)

        # To upload the file to the bucket
        blob.upload_from_file(file)

        # To inform the user that their file is being uploaded successfully
        print(f"File {destination_blob_name} uploaded to {bucket_name}.")
        st.success(f"File uploaded successfully: {destination_blob_name}")

    except Exception as e:
        # I am printing error message such that the user knows if upload fails
        st.error(f"Error uploading file: {e}")
        return None

def load_model(model_name):
    """Loading pretrained model, then print error message if the model does not load"""
    try:
        if model_name == 'ResNet50':
            return ResNet50(weights='imagenet')
        elif model_name == 'VGG16':
            return VGG16(weights='imagenet')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_image(model, img_path, model_name):
    """This function is to preprocesses the image, makes predictions
    using the model, and returns the top 5 predicted classes and their probabilities.
    Input of this function is model (the loaded pretrained model), img_path, and model_name.
    model_name is a string telling us which model is being used.
    We will print error message if we cannot predict the image."""
    try:
        # First we load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocessing steps // different for each model
        if model_name == 'ResNet50':
            preprocessed_img = resnet_preprocess(img_array)
            predictions = model.predict(preprocessed_img)
            decoded_predictions = resnet_decode(predictions, top=5)
        else:  # this is for VGG16
            preprocessed_img = vgg_preprocess(img_array)
            predictions = model.predict(preprocessed_img)
            decoded_predictions = vgg_decode(predictions, top=5)

        return decoded_predictions
    except Exception as e:
        st.error(f"Error predicting image: {e}")
        return None

def plot_predictions(predictions):
    """Create a pie chart of top predictions"""
    try:
        labels = [pred[1] for pred in predictions[0]]
        scores = [float(pred[2]) * 100 for pred in predictions[0]]

        plt.figure(figsize=(10, 6))
        plt.pie(scores, labels=labels, autopct='%1.1f%%')
        plt.title('Top 5 Prediction Probabilities')

        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Error creating prediction plot: {e}")
        return None

def download_test_image(url):
    """Download test image from URL"""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Generate a unique filename
            file_extension = url.split('.')[-1]
            filename = f"test_image_{uuid.uuid4()}.{file_extension}"

            # Save the image
            with open(filename, 'wb') as f:
                f.write(response.content)
            return filename
        else:
            st.error(f"Failed to download image from {url}")
            return None
    except Exception as e:
        st.error(f"Error downloading test image: {e}")
        return None

def main():
    st.title('Image Classification with Pre-trained CNN Models')

    # Sidebar for model selection
    st.sidebar.header('Model Configuration')
    model_choice = st.sidebar.selectbox(
        'Select Pre-trained Model',
        ['ResNet50', 'VGG16']
    )

    # Image Upload
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Generate a unique filename
        file_extension = uploaded_file.name.split('.')[-1]
        unique_filename = f"uploads/{uuid.uuid4()}.{file_extension}"

        # Upload to Google Cloud Storage
        gcs_path = upload_blob('cpsc-4820-image-upload', unique_filename, uploaded_file)

        # Save uploaded file locally for prediction
        with open('temp_image.jpg', 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Predict button
        if st.button('Predict Image'):
            # Load selected model
            model = load_model(model_choice)

            if model:
                # Make predictions
                predictions = predict_image(model, 'temp_image.jpg', model_choice)

                if predictions:
                    # Display text predictions
                    st.subheader('Top 5 Predictions')
                    for i, (id, label, score) in enumerate(predictions[0], 1):
                        st.write(f"{i}. {label} ({score*100:.2f}%)")

                    # Plot predictions
                    pred_plot = plot_predictions(predictions)
                    if pred_plot:
                        st.image(pred_plot, caption='Prediction Probabilities', use_column_width=True)

    # sidebar header. also initializing dictionary of images
    st.sidebar.header('Test Images')
    test_images = {
        'Cat': 'https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg',
        'Computer': 'https://editmicro.co.za/wp-content/uploads/2014/04/Desktop-Computer-by-edit-micro-768x510.jpg',
        'Building': 'https://www.shutterstock.com/shutterstock/photos/2501530247/display_1500/stock-photo-new-modern-block-of-flats-in-green-area-residential-apartment-with-flat-buildings-exterior-luxury-2501530247.jpg',
        'Baby': 'https://kendamil.com/cdn/shop/articles/iStock-1017721326.jpg',
        'Car': 'https://evo.ca/-/media/evo/images/home-page/lifestyle/car1-static.png'
    }

    # sidebar showing list of images from the internet for the user to select
    selected_test_image = st.sidebar.selectbox(
        'Select a Test Image',
        list(test_images.keys())
    )

    # sidebar button 'Predict Test Image'
    if st.sidebar.button('Predict Test Image'):
        # download the test image first
        test_image_path = download_test_image(test_images[selected_test_image])

        if test_image_path:
            # if the image is downloaded, then we load the selected model
            model = load_model(model_choice)

            if model:
                # the model makes prediction
             predictions = predict_image(model, test_image_path, model_choice)

            if predictions:
                    # we display the test image
                    st.image(test_image_path, caption=f'Test Image: {selected_test_image}', use_column_width=True)

                    # this is to display the top 5 predictions and their score
                    st.subheader('Top 5 Predictions')
                    for i, (id, label, score) in enumerate(predictions[0], 1):
                        st.write(f"{i}. {label} ({score*100:.2f}%)")

                    # this is to show the pie chart of the predictions
                    pred_plot = plot_predictions(predictions)
                    if pred_plot:
                        st.image(pred_plot, caption='Prediction Probabilities', use_column_width=True)

if __name__ == '__main__':
    main()

























