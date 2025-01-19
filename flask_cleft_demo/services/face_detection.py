import cv2
import dlib

# 加载人脸检测器和 landmarks 预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("flask_cleft_demo\services\shape_predictor_68_face_landmarks.dat")

def detect_faces(image):
    """
    检测人脸并绘制 landmarks
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # 在图像上绘制

    return faces, image
