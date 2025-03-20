from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import base64
import numpy as np
import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Define harmful objects
harmful_objects = ["knife", "gun", "scissors"]

@csrf_exempt
def detect_objects(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            frame_data = data.get("frame")

            if not frame_data:
                return JsonResponse({"error": "No frame data received"}, status=400)

            # Decode Base64 frame
            frame_data = base64.b64decode(frame_data.split(",")[1])
            np_frame = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

            # Run YOLO detection
            results = model(frame)
            #print(results)
            detected_objects = []

            for result in results:
                for obj in result.boxes.data:
                    x1, y1, x2, y2, conf, cls = obj.tolist()
                    label = model.names[int(cls)]

                    # If the object is harmful, add to list
                    if label in harmful_objects:
                        detected_objects.append(label)

            return JsonResponse({"detected": detected_objects})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"message": "Only POST requests allowed."},
 status=400)
def index(request):
    return render(request, 'index.html')
