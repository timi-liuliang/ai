from ultralytics import YOLO

model = YOLO("model/yolov8m-cls.pt")
model.predict(
    source="input/SORRENTO Italy Walking Tour In 4K _ Watch SORRENTO in 10 Minutes.mp4", 
    save=False, 
    conf=0.5, 
    save_txt=False, 
    show=True)