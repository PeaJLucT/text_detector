import os
import cv2
import torch
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageOps, ImageFont
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re
from functions import detect

# НАСТРОЙКИ МОДЕЛЕЙ 
YOLO_MODEL_PATH = './segmentation best weight/best_1.pt'  # Путь к YOLO модели 1
YOLO_MODEL_PATH_2 = './segmentation best weight/best_4.pt'  # Путь к YOLO модели 2

TROCR_MODEL_PATH = "./text_recognition_model/model"
TROCR_BATCH_SIZE = 16
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# Папки
example_images = 'text_detector/examples'        # Где хранятся изображения для чтения
images_output = 'final_results'                  # Куда сохранять картинки с текстом
words_output_dir = ''                            # Куда сохранять нарезанные слова/оставить '' если не требуется
text_file_output = 'full_text1.txt'              # Куда сохранить весь текст


def preprocess_word_crop(crop: Image.Image) -> Image.Image:
    """Удаляет горизонтальные линии и повышает контраст перед TrOCR."""
    gray = cv2.cvtColor(np.array(crop.convert("RGB")), cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    h, w = gray.shape
    if w >= 12 and h >= 6:
        k_w = max(15, int(w * 0.6))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
        lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

        _, mask = cv2.threshold(lines, 15, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(mask) > 0:
            gray = cv2.inpaint(gray, mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)

    gray = clahe.apply(gray)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    return Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))


def load_models(trocr_model_path=TROCR_MODEL_PATH):
    """
    Загружает обе модели yolo и trOcr в память один раз при старте 
    """
    print(f"Загрузка моделей на {DEVICE}")

    try:
        yolo_model = YOLO(YOLO_MODEL_PATH, task='detect')
        print(f"✅ Первая YOLO модель загружена из {YOLO_MODEL_PATH}")
        yolo_model.to(DEVICE) 
    except Exception as e:
        print(f"❌ Ошибка загрузки первой YOLO модели: {e}")
        raise
    try:
        yolo_model_2 = YOLO(YOLO_MODEL_PATH_2, task='detect')
        print(f"✅ Вторая YOLO модель загружена из {YOLO_MODEL_PATH_2}")
        yolo_model_2.to(DEVICE)
    except Exception as e:
        print(f"❌ Ошибка загрузки второй YOLO модели: {e}")
        raise
    

    # TrOCR
    try:
        config_path = os.path.join(trocr_model_path, "config.json")
        weights_path = os.path.join(trocr_model_path, "model.safetensors")
        if not os.path.isdir(trocr_model_path) or not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Папка модели TrOCR не найдена: {os.path.abspath(trocr_model_path)}"
            )
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Файл весов TrOCR не найден: {os.path.abspath(weights_path)}"
            )

        print(f"Загрузка TrOCR из {os.path.abspath(trocr_model_path)}")
        processor = TrOCRProcessor.from_pretrained(trocr_model_path, local_files_only=True)
        trocr_model = VisionEncoderDecoderModel.from_pretrained(
            trocr_model_path, local_files_only=True
        )
        trocr_model.to(DEVICE)
        trocr_model.eval()
        print("✅ TrOCR модель загружена")
    except Exception as e:
        print(f"❌ Ошибка загрузки TrOCR модели: {e}")
        print("Проверьте папку text_recognition_model/model (config.json, model.safetensors, tokenizer).")
        raise
    
    print("Модели успешно загружены")
    return yolo_model, yolo_model_2,  processor, trocr_model

def detect_and_read(yolo_model, yolo_model_2, processor, trocr_model, image_path, conf=0.3, output_folder=''):
    '''
    Находит слова и читает их
    
    :param yolo_model: Модель YOLOv8
    :param yolo_model_2: Модель YOLOv8
    :param image_path: Путь до изображения для чтения текста на нем
    :param conf: Уверенность модели 
    :param output_folder: 
    '''

    try:
        orig_img = Image.open(image_path).convert('RGB')
        orig_img = ImageOps.exif_transpose(orig_img)
        print(f'Изображение успешно загружено из {image_path}')
    except Exception as e:
        print(f"Ошибка при открытии изображения: {e}")
        return [], None, ''

    finded_images, image_with_boxes, count_words, sorted_boxes = detect(
        model_path = yolo_model,
        model_path_2= yolo_model_2, 
        conf = conf, 
        threshold_value=0.9,
        image_path= image_path,
        output_folder=words_output_dir
        )
    draw = ImageDraw.Draw(image_with_boxes)
    try:
        font = ImageFont.truetype("arial.ttf", 40)  
    except IOError:
        font = ImageFont.load_default()

    if count_words == 0:
        print("⚠️ YOLO не нашел слов на изображении. Попробуйте уменьшить параметр уверенности.")
        return [], orig_img, ""

    detected_data = list()

    # РАСПОЗНАВАНИЕ TrOCR (пакетами)
    print(f"Начинаю распознавание {count_words} слов (batch={TROCR_BATCH_SIZE})...")
    padding = 5
    width, height = orig_img.size
    crops = []
    boxes_coords = []

    for box in sorted_boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1_p = max(0, x1 - padding)
        y1_p = max(0, y1 - padding)
        x2_p = min(width, x2 + padding)
        y2_p = min(height, y2 + padding)
        crops.append(preprocess_word_crop(orig_img.crop((x1_p, y1_p, x2_p, y2_p))))
        boxes_coords.append([x1, y1, x2, y2])

    word_texts = []
    for batch_start in range(0, len(crops), TROCR_BATCH_SIZE):
        batch_crops = crops[batch_start:batch_start + TROCR_BATCH_SIZE]
        pixel_values = processor(batch_crops, return_tensors="pt").pixel_values.to(DEVICE)

        with torch.no_grad():
            generated_ids = trocr_model.generate(
                pixel_values,
                max_length=50,
                num_beams=6,
                early_stopping=True,
            )

        batch_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        word_texts.extend(
            re.sub(r'[^а-яА-ЯёЁ0-9\.,\-\!\? ]', '', text)
            for text in batch_texts
        )

        processed = min(batch_start + len(batch_crops), len(crops))
        print(f"  Обработано {processed}/{len(crops)} слов...")

    full_text_list = []
    for word_text, (x1, y1, x2, y2) in zip(word_texts, boxes_coords):
        detected_data.append({'text': word_text, 'box': [x1, y1, x2, y2]})
        full_text_list.append(word_text)

        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        label = word_text if word_text.strip() else "???"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(
            (x1, y1 - (text_bbox[3] - text_bbox[1]) - 5, x1 + (text_bbox[2] - text_bbox[0]) + 5, y1),
            fill="red",
        )
        draw.text((x1, y1 - (text_bbox[3] - text_bbox[1]) - 5), label, fill="white", font=font)
    
    final_text = " ".join(full_text_list).strip()
    recognized_count = sum(1 for word in full_text_list if word.strip())
    print(f"Распознано слов: {recognized_count}/{len(sorted_boxes)}")
    
    if not final_text:
        print("⚠️ Все слова найдены, но текст не распознан. Возможна проблема с моделью TrOCR.")
    
    return finded_images, image_with_boxes, final_text, detected_data

if __name__ == '__main__':
    yolo, yolo_2, processor, trocr = load_models()
    os.makedirs(images_output, exist_ok=True)

    open(text_file_output, 'w', encoding='utf-8').close()

    for image_file in os.listdir(example_images):  
        image_path = os.path.join(example_images, image_file)
        
        print(f"\nОбработка: {image_file}")
        
        _, img_result, text_result, detected_data = detect_and_read(
            yolo, yolo_2, processor, trocr, 
            image_path, 
            conf=0.4, # Порог уверенности для YOLO
            output_folder=words_output_dir
        )
        
        # Сохраняем картинку с подписанными найденными словами
        if img_result:
            save_path = os.path.join(images_output, f'recognized_{image_file}')
            img_result.save(save_path)
            print(f"Картинка сохранена: {save_path}")
        
        # Сохраняем текст в файл
        if text_result:
            print(f"Текст:\n{text_result[:100]}...") 
            with open(text_file_output, 'a', encoding='utf-8') as f:
                f.write(f"=== {image_file} ===\n")
                f.write(text_result + "\n\n")