from ultralytics import YOLO

model = YOLO("model/yolov8x-seg.pt")
model.predict(
    source="input/[4K] 10 minutes Walk _ Neighbourhood Walk _ Sunny Day _ 日本 _ Japan Walk _ ASMR Walk.mp4", 
    save=False, 
    conf=0.5, 
    save_txt=False, 
    show=True)

#model.export(format="onnx")