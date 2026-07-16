import  gradio as gr
from ultralytics import YOLO

YOLO("yolo11n.pt")

model = YOLO("best.pt")
def detect_objects(image):
    results = model.predict(image, save=True)
    return results[0].plot()

demo = gr.Interface(fn=detect_objects, inputs=gr.Image(type="numpy"), outputs=gr.Image(type="numpy"))

demo.launch()