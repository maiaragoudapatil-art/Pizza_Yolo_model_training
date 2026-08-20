from ultralytics import YOLO

model = YOLO(r"C:\Users\VijaySegunasi\runs\segment\train-4\weights\best.pt")

results = model.predict(
    source=r"C:\Users\VijaySegunasi\pizza_yolo_model\test_images",
    save=True,
    conf=0.25
)

with open("prediction_report.txt", "w") as f:
    for r in results:
        f.write(f"\nImage: {r.path}\n")

        if r.boxes is not None:
            for box in r.boxes:
                cls = int(box.cls)
                conf = float(box.conf)

                class_name = model.names[cls]

                f.write(
                    f"Detected: {class_name}, "
                    f"Confidence: {conf:.2f}\n"
                )
