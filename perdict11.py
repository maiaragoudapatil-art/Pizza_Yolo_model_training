from ultralytics import YOLO

model = YOLO(
    r"C:\Users\VijaySegunasi\pizza_yolo_model\pizza_yolo11n_seg\weights\best.pt"
)

results = model.predict(
    source=r"C:\Users\VijaySegunasi\pizza_yolo_model\test_images",
    save=True,
    project=r"C:\Users\VijaySegunasi\pizza_yolo_model",
    name="prediction_results_yolo11",
    conf=0.25
)

report_path = r"C:\Users\VijaySegunasi\pizza_yolo_model\prediction_report_yolo11.txt"

with open(report_path, "w") as f:
    f.write(f"Total Images Tested: {len(results)}\n\n")

    for i, r in enumerate(results, start=1):
        f.write(f"Image {i}: {r.path}\n")

        if r.boxes is not None:
            for box in r.boxes:
                cls = int(box.cls)
                confidence = float(box.conf)

                class_name = model.names[cls]

                f.write(
                    f"  - Detected: {class_name}, "
                    f"Confidence: {confidence:.2%}\n"
                )

        f.write("\n")

print("Prediction complete for YOLO11n!")
print("Images saved in: prediction_results_yolo11")
print("Report saved as: prediction_report_yolo11.txt")