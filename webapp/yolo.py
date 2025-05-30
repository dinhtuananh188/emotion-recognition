from ultralytics import YOLO

model = YOLO("yolo11n.pt")  

results = model.train(data="Tu_lam_v2/data.yaml", epochs=250, imgsz=640)