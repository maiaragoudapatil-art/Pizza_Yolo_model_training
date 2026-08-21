from ultralytics import YOLO

# YOLOv8n
model8 = YOLO(
    r"C:\Users\VijaySegunasi\pizza_yolo_model\pizza_segmentation_new\weights\best.pt"
)

# YOLO11n
model11 = YOLO(
    r"C:\Users\VijaySegunasi\pizza_yolo_model\pizza_yolo11n_seg\weights\best.pt"
)

print("\n========== YOLOv8n ==========")
metrics8 = model8.val(
    data=r"C:\Users\VijaySegunasi\pizza_yolo_model\data.yaml"
)

print("\n========== YOLO11n ==========")
metrics11 = model11.val(
    data=r"C:\Users\VijaySegunasi\pizza_yolo_model\data.yaml"
)

print("\n========== COMPARISON ==========")

print("YOLOv8n:")
print("Box mAP50-95 :", metrics8.box.map)
print("Mask mAP50-95:", metrics8.seg.map)

print("\nYOLO11n:")
print("Box mAP50-95 :", metrics11.box.map)
print("Mask mAP50-95:", metrics11.seg.map)