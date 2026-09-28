from ultralytics import YOLO

model = YOLO(
    r"C:\Users\VijaySegunasi\runs\segment\train-4\weights\best.onnx"
)

results = model.predict(
    source=r"C:\Users\VijaySegunasi\pizza_yolo_model\test_images",
    save=True,
    project=r"C:\Users\VijaySegunasi\pizza_yolo_model",
    name="onnx_prediction_results",
    conf=0.25
)

print("ONNX model prediction successful!")