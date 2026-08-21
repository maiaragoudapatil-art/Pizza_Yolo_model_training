from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

model.train(
    data="data.yaml",
    epochs=10,
    batch=8,
    imgsz=640,
    project="runs",
    name="pizza_segmentation_new"
)
#C:\Users\VijaySegunasi\runs\segment\train-4
