from fastai.vision.all import *
import gradio as gr
import cv2


learn = load_learner("model.pkl")
print("vocab", learn.dls.vocab)

categories = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")


def preprocess_webcam_image(img_array):
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Face detection using Haar cascades
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # If no face is detected, return the original image (or handle this case appropriately)
    if len(faces) == 0:
        print("No face detected")
        return gray / 255.0

    # Assuming only one face is detected, crop it
    x, y, w, h = faces[0]
    cropped_face = gray[y : y + h, x : x + w]

    # Resize to match training data dimensions (e.g., 224x224)
    resized_face = cv2.resize(cropped_face, (224, 224))

    # Normalize if necessary (e.g., scale pixel values to [0, 1])
    normalized_face = resized_face / 255.0

    return normalized_face


def classify_img(img):
    processed_img = preprocess_webcam_image(img)
    pred, idx, probs = learn.predict(processed_img)

    # Debugging print statement
    print(f"Predicted: {pred}, Probs: {probs}")

    return {category: prob.item() for category, prob in zip(categories, probs)}


image = gr.inputs.Image(shape=(1000, 1000))
label = gr.outputs.Label()
# examples = ["angry.jpg", "sad.jpg"]

iface = gr.Interface(
    fn=classify_img,
    inputs=gr.Image(source="webcam", streaming=True),
    outputs=label,
    live=True,
)
iface.launch(inline=False)
