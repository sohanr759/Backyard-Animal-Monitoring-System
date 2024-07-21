import cv2
import numpy as np
import tensorflow as tf
import time

# Load the MobileNetV2 model pre-trained on ImageNet
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Create a function to preprocess an image for inference
def preprocess_image(image):
    img = tf.image.resize(image, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

# Create a function to perform animal detection on an image
def detect_animal(image):
    img = preprocess_image(image)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    predictions = model.predict(img)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]
    top_prediction = decoded_predictions[0]
    label, score = top_prediction[1], top_prediction[2]
    return label, score

# Open the webcam
cap = cv2.VideoCapture(0)

# Set the duration for capturing frames (in seconds)
capture_duration = 10
start_time = time.time()

while time.time() - start_time < capture_duration:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Detect animals in the frame
    label, score = detect_animal(frame)
    
    # Display the result on the frame
    result_text = f"Animal: {label} (Score: {score:.2f})"
    cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Animal Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

print(result_text)


from twilio.rest import Client

# Your Twilio Account SID and Authentication Token
account_sid = 'ACca712c5fa2169d03915f9c5158f1d7c2'
auth_token = '744d41b8e0eb2a3287d71fa9902cc8e9'

# Create a Twilio client
client = Client(account_sid, auth_token)

# Your Twilio phone number (must be purchased from Twilio)
twilio_phone_number = '+12345420437'

# The recipient's phone number (in E.164 format, e.g., +1234567890)
recipient_phone_number = '+919346395037 '

# The message you want to send
message = '''An animal has been detected in the backyard. It appears to be a '''+ label + '''. Please be cautious and avoid approaching the animal. Ensure that pets are kept indoors for safety.'''

try:
    # Send the SMS
    message = client.messages.create(
        body=message,
        from_=twilio_phone_number,
        to=recipient_phone_number
    )

    print(f"Message sent with SID: {message.sid}")
except Exception as e:
    print(f"An error occurred: {str(e)}")
