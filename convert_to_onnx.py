import torch

# Muat model YOLOv5 (misalnya 'yolov5s')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Buat input dummy
dummy_input = torch.randn(1, 3, 640, 640)

# Konversi model ke format ONNX
torch.onnx.export(model, dummy_input, "yolov5.onnx", opset_version=12)
