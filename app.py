import os
import gradio as gr
from ultralytics import YOLO

# Explicitly direct Ultralytics away from restricted /opt/render/ path
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

model = YOLO("best.pt")

def pred_image(image):
    results = model.predict(image)
    # Convert BGR (OpenCV default) to RGB for Gradio display
    res_bgr = results[0].plot()
    res_rgb = res_bgr[:, :, ::-1]
    return res_rgb

app = gr.Interface(
    fn=pred_image, 
    inputs="image", 
    outputs="image"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.launch(
        server_name="0.0.0.0",
        server_port=port
    )

model = YOLO("best.pt")

def pred_image(image):
    img = model.predict(image)
    return img[0].plot()


app= gr.Interface(fn = pred_image, inputs = 'image', outputs = "image" )
app.launch(
    server_name="0.0.0.0",
    server_port=int(_import_("os").environ.get("PORT", 10000))
)
