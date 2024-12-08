# Exercise Video Analysis API
This API analyzes exercise videos to determine the type of exercise being performed and provides metrics on form and muscle engagement.

## Description
The Exercise Video Analysis API uses YOLOv5 for object detection and Mediapipe for pose estimation to analyze exercise videos. It detects exercises such as overhead squats, glute bridges, back lunges, and box jumps, and provides metrics such as angles and correctness of movements. The processed videos are stored in Google Cloud Storage and annotated with performance metrics.

## Technologies Used
1. FastAPI: Python web framework used for building the API endpoints.
2. YOLOv5: State-of-the-art deep learning model for real-time object detection.
3. Mediapipe: Framework by Google Research for perception pipelines, including pose estimation.
4. OpenCV (cv2): Library for computer vision tasks such as image and video processing.
5. Google Cloud Platform (GCP):
6. Cloud Storage: Used for storing input and output videos securely.
7. Cloud Run: Serverless compute platform for deploying and scaling containerized applications.
8. Python Libraries: requests, numpy, ultralytics for handling HTTP requests, numerical operations, and integrating with YOLOv5, respectively.



https://github.com/user-attachments/assets/d2cd188e-ce82-4865-93d2-46ec6b27affb



https://github.com/user-attachments/assets/23e7f49e-c467-4de8-b23d-759b688fbe05


