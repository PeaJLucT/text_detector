import os
import cv2
import torch
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageOps, ImageFont
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re

# НАСТРОЙКИ МОДЕЛЕЙ 
YOLO_MODEL_PATH = 'weights/last.pt'  # Путь к YOLO
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
    Загружает обе модели в память один раз при старте
    """
    print(f"Загрузка моделей на {DEVICE}")
    
    # YOLO
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH, task='detect')
        print(f"✅ YOLO модель загружена из {YOLO_MODEL_PATH}")
    except Exception as e:
        print(f"❌ Ошибка загрузки YOLO модели: {e}")
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
    return yolo_model, processor, trocr_model

def sort_boxes(boxes):
    """Сортирует рамки: строки сверху-вниз, слова слева-направо"""
    if not boxes:
        return []

    if hasattr(boxes, 'data'):
        boxes_data = boxes.data.cpu().numpy()
    else:
        boxes_data = boxes 
        
    heights = boxes_data[:, 3] - boxes_data[:, 1]
    avg_height = np.mean(heights)
    
    # Коэффициент 0.6-0.7 для создания строк
    sorted_boxes = sorted(boxes, key=lambda b: (b.xyxy[0][1].item() // (avg_height * 0.6), b.xyxy[0][0].item()))
    return sorted_boxes

def preprocess_line_image(image):
    """
    Предобработка изображения строки для улучшения распознавания
    
    Args:
        image: PIL Image
        
    Returns:
        PIL Image с улучшенным контрастом и нормализацией
    """
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
    enhanced = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
    
    mean_intensity = np.mean(enhanced)
    if mean_intensity < 127:
        enhanced = cv2.bitwise_not(enhanced)
    
    min_height = 32
    if enhanced.shape[0] < min_height:
        scale_factor = min_height / enhanced.shape[0]
        new_width = int(enhanced.shape[1] * scale_factor)
        enhanced = cv2.resize(enhanced, (new_width, min_height), interpolation=cv2.INTER_CUBIC)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
      
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(enhanced_rgb)

def detect_and_read(yolo_model, processor, trocr_model, image_path, conf=0.5, output_folder=''):
    '''
    Находит слова и читает их
    
    :param yolo_model: Модель YOLOv8
    :param image_path: Путь до изображения для чтения текста на нем
    :param conf: Уверенность модели 
    :param output_folder: 
    '''
    try:
        orig_img = Image.open(image_path).convert('RGB')
        orig_img = ImageOps.exif_transpose(orig_img)
    except Exception as e:
        print(f"Ошибка открытия файла: {e}")
        return [], None, ""

    # ДЕТЕКЦИЯ YOLO
    img_gray_for_yolo = ImageOps.grayscale(orig_img).convert('RGB')
    
    results = yolo_model.predict(img_gray_for_yolo, imgsz=1280, conf=conf, verbose=False)
    result = results[0]
    
    sorted_boxes = sort_boxes(result.boxes)
    print(f"Найдено слов: {len(sorted_boxes)}")

    if len(sorted_boxes) == 0:
        print("⚠️ YOLO не нашел слов на изображении. Попробуйте уменьшить параметр уверенности.")
        return [], orig_img, ""

    image_with_boxes = orig_img.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    
    try:    
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    finded_images = []
    full_text_list = []

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    if output_folder:
        current_img_output = os.path.join(output_folder, image_name)
        os.makedirs(current_img_output, exist_ok=True)

    # РАСПОЗНАВАНИЕ TrOCR
    print(f"Начинаю распознавание {len(sorted_boxes)} слов...")
    for i, box in enumerate(sorted_boxes):
        # Координаты
        c = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = c[0], c[1], c[2], c[3]

            # №№№№№№№№№№
        padding = 10              
        width, height = orig_img.size
        x1_p = max(0, x1 - padding)
        y1_p = max(0, y1 - padding)
        x2_p = min(width, x2 + padding)
        y2_p = min(height, y2 + padding)
        crop_image = orig_img.crop((x1_p, y1_p, x2_p, y2_p))
            # №№№№№№№№№№


        # crop_image = orig_img.crop((x1, y1, x2, y2))
        if output_folder:
            crop_image.save(os.path.join(current_img_output, f'{i}.jpg'), quality=100)
        finded_images.append(crop_image)

        # processed_crop = preprocess_line_image(crop_image      # №№№№№№№№№№# №№№№№№№№№№
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
        full_text_list.append(word_text)
        
        if (i + 1) % 10 == 0:
            print(f"  Обработано {i + 1}/{len(sorted_boxes)} слов...")


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
    
    return finded_images, image_with_boxes, final_text

if __name__ == '__main__':
    yolo, processor, trocr = load_models()
    os.makedirs(images_output, exist_ok=True)

    open(text_file_output, 'w', encoding='utf-8').close()

    for image_file in os.listdir(example_images):  
        image_path = os.path.join(example_images, image_file)
        
        print(f"\nОбработка: {image_file}")
        
        _, img_result, text_result = detect_and_read(
            yolo, processor, trocr, 
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