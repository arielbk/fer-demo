from fastai.vision.all import *
import gradio as gr
import cv2


learn = load_learner("model.pkl")
print("vocab", learn.dls.vocab)


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

    # Convert back to a 3-channel image
    processed_img_rgb = np.stack([normalized_face] * 3, axis=-1)
    processed_img_rgb = (processed_img_rgb * 255).astype(np.uint8)

    return processed_img_rgb


def classify_img(img):
    processed_img = preprocess_webcam_image(img)
    pred, idx, probs = learn.predict(processed_img)

    parsed_probs = dict(zip(learn.dls.vocab, map(float, probs)))
    # Debugging print statement
    print(parsed_probs)

    # Return both the prediction dictionary and the processed image
    return (
        processed_img,
        parsed_probs,
    )


# image = gr.inputs.Image(shape=(192, 192))
image = gr.Image(source="webcam", shape=(224, 224), streaming=True, interactive=True)
label = gr.outputs.Label()
examples = ["angry.jpg", "happy.jpg"]

iface = gr.Interface(
    fn=classify_img,
    inputs=image,
    outputs=[
        gr.outputs.Image(type="numpy", label="Processed Image").style(height=224),
        gr.outputs.Label(),
    ],
    examples=examples,
)
iface.dependencies[0]["show_progress"] = False
iface.launch(inline=False)
