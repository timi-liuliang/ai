from ultralytics import YOLO

model = YOLO("yolov8m.pt")
model.predict(source="input/1.png", save=True, conf=0.5, save_txt=True)

model.export(format="onnx")