from ultralytics import YOLO

model = YOLO(
    r"C:\Users\VijaySegunasi\runs\segment\train-4\weights\best.pt"
)

model.export(format="onnx")