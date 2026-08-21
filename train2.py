from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")

model.train(
    data="data.yaml",
    epochs=15,
    batch=8,
    imgsz=640,
    project="runs",
    name="pizza_yolo11n_seg"
)