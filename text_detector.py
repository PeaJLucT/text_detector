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

TROCR_MODEL_NAME = "cyrillic-trocr/trocr-handwritten-cyrillic" # Путь к TrOCR 
# TROCR_MODEL_NAME = "kaz-v/trocr-handwritten-russian"         
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# Папки
example_images = 'text_detector/examples'        # Где хранятся изображения для чтения
images_output = 'final_results'                  # Куда сохранять картинки с текстом
words_output_dir = ''                            # Куда сохранять нарезанные слова/оставить '' если не требуется
text_file_output = 'full_text1.txt'              # Куда сохранить весь текст



def load_models():
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
        print("Загрузка TrOCR модели с HuggingFace (может занять время при первом запуске)...")
        processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_NAME)
        trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME)
        trocr_model.to(DEVICE)
        trocr_model.eval()
        print("✅ TrOCR модель загружена")
    except Exception as e:
        print(f"❌ Ошибка загрузки TrOCR модели: {e}")
        print("Проверьте интернет-соединение. Модель загружается с HuggingFace.")
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

    full_text_list = []

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    if output_folder:
        current_img_output = os.path.join(output_folder, image_name)
        os.makedirs(current_img_output, exist_ok=True)
    
    detected_data = list()

    # РАСПОЗНАВАНИЕ TrOCR
    print(f"Начинаю распознавание {count_words} слов...")
    for i, box in enumerate(sorted_boxes):
        c = box.xyxy[0].tolist()
        x1, y1, x2, y2 = c[0], c[1], c[2], c[3]

        padding = 5              
        width, height = orig_img.size
        x1_p = max(0, x1 - padding)
        y1_p = max(0, y1 - padding)
        x2_p = min(width, x2 + padding)
        y2_p = min(height, y2 + padding)
        crop_image = orig_img.crop((x1_p, y1_p, x2_p, y2_p))
        
        processed_crop = crop_image
        pixel_values = processor(processed_crop, return_tensors="pt").pixel_values.to(DEVICE)
        
        with torch.no_grad():
            generated_ids = trocr_model.generate(
                pixel_values,
                max_length=50,    # Длина для слова
                num_beams=6,      # Качество поиска
                early_stopping=True
            )
        
        word_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        word_text = re.sub(r'[^а-яА-ЯёЁ0-9\.,\-\!\? ]', '', word_text)  # №№№№№№№№№№
        
        detected_data.append({'text':word_text, 'box':[x1, y1, x2, y2]})
        
        full_text_list.append(word_text)
        
        if (i + 1) % 10 == 0:
            print(f"  Обработано {i + 1}/{len(finded_images)} слов...")


        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        if word_text.strip():
            label = word_text
        else:
            label = "???"
            
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle((x1, y1 - (text_bbox[3]-text_bbox[1]) - 5, x1 + (text_bbox[2]-text_bbox[0]) + 5, y1), fill="red")
        draw.text((x1, y1 - (text_bbox[3]-text_bbox[1]) - 5), label, fill="white", font=font)
    
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