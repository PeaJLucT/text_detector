import cv2
import os
import time

"""
разделяет видео путь к которому хранится в video_path 
на изображения каждый кадр (хранящийся в frame_interval)
"""

# Путь к видео
video_path = 'video for split.mp4'
# Папка для сохранения изображений после разделения
output_folder = 'video_images'
# Интервал между кадрами (каждый n-й кадр будет сохранен)
frame_interval = 15

os.makedirs(output_folder, exist_ok=True)

# Открытие видеофайла
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Ошибка открытия видеофайла: {video_path}")
    exit()

frame_count = 0
saved_count = 0

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Ширина видео: {width}, Высота кадра: {height}")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        # Получение текущего времени в виде временной метки
        timestamp = int(time.monotonic() * 100)  # Используем миллисекунды для большей точности
        output_path = os.path.join(output_folder, f'{timestamp}_frame_{saved_count:05d}.jpg')
        cv2.imwrite(output_path, frame)
        print(f"Сохранено: {output_path}, {saved_count} кадр")
        saved_count += 1

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("Разделение видео на фотографии завершено.")