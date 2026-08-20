from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

model.train(
    data="data.yaml",
    epochs=10,
    batch=8,
    imgsz=640
)
#C:\Users\VijaySegunasi\runs\segment\train-4