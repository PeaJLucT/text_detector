from PIL import Image, ImageDraw, ImageOps, ImageFont, ImageEnhance
import PIL
import os
from ultralytics import YOLO
import numpy as np
import cv2


def sort_boxes(boxes):
    """
    Сортирует рамки в порядке чтения: сверху-вниз, слева-направо
    """
    if not boxes:
        return []
    # определяем среднюю высоту рамок, чтобы задать допуск для одной строки
    boxes_data = boxes.data.cpu().numpy() 
    heights = boxes_data[:, 3] - boxes_data[:, 1]
    avg_height = np.mean(heights)
    # print(f'Средний допуск строки - {avg_height}')
    
    # Сортируем рамки по строкам
    sorted_boxes = sorted(boxes, key=lambda b: (b.xyxy[0][1].item() // (avg_height * 0.7), b.xyxy[0][0].item()))
    
    return sorted_boxes

def pre_process_image_for_model(pil_image):
    '''
    Использует умное улучшение контраста (CLAHE), чтобы вытащить карандаш,
    не ломая при этом ручку.
    '''
    img_np = np.array(pil_image.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)

    # gamma = 0.8
    # enhanced_gray = np.array(255 * (enhanced_gray / 255) ** (1 / gamma), dtype='uint8')

    enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(enhanced_rgb)


def detect(model_path: str, image_path: str, draw_graphs = False, conf = 0.3, output_folder = '') -> tuple[list[PIL.Image.Image], PIL.Image.Image]:
    '''
    Возвращает список обнаруженных и вырезанных слов с изображения

    Args:
        model_path (str): Путь до лучшей модели
        image_path (str): Путь до изображения *.jpg | *.png
        draw_graphs (bool): True - если нужно вывести в output изображение оригинальное и с маской найденных слов
        conf (float): Уверенность модели для записи маски как правильной
        output_folder (str): Путь до папки, в которую будут сохранены все вырезанные найденные слова с изображения
    Returns:
        list: [ [Список найденных изображений] , оригинальное изображение с маской детекции]
    '''
    
    try:
        model = YOLO(model_path, task='detect')
        print(f"Лучшая модель успешно загружена из {model_path}")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return [], None

    try:
        orig_img = Image.open(image_path).convert('RGB')
        orig_img = ImageOps.exif_transpose(orig_img)
        print(f'Изображение успешно загружено из {image_path}')
    except Exception as e:
        print(f"Ошибка при открытии изображения: {e}")
        return [], None
    
    # Детектим слова
    img_gray_for_model = ImageOps.grayscale(orig_img).convert('RGB')
    
    # img_gray_for_model = pre_process_image_for_model(orig_img)
    results = model.predict(img_gray_for_model, imgsz=1280, conf=conf, verbose=False)
    result = results[0]

    sorted_boxes = sort_boxes(result.boxes)
    
    image_with_boxes = orig_img.copy()
    draw = ImageDraw.Draw(image_with_boxes)

    try:    
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    # список найденных слов 
    finded_images = list()

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    

    if output_folder:
        current_img_output = os.path.join(output_folder, image_name)
        #папка для сохранения изображений слов
        os.makedirs(current_img_output, exist_ok=True)

    for i, box in enumerate(sorted_boxes):
        c = box.xyxy[0].cpu().numpy().astype(int) # x1, y1, x2, y2
        x1, y1, x2, y2 = c[0], c[1], c[2], c[3]

        confid = float(box.conf[0])
        
        # Вырезаем слово
        crop_image = orig_img.crop((x1, y1, x2, y2))
        if output_folder:
            crop_image.save(os.path.join(current_img_output, f'{i}.jpg'), quality = 100)
        finded_images.append(crop_image)

        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)

        label = f"{confid:.2f}"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle((x1, y1 - (text_bbox[3]-text_bbox[1]), x1 + (text_bbox[2]-text_bbox[0]), y1), fill="red")
        draw.text((x1, y1 - (text_bbox[3]-text_bbox[1])), label, fill="white", font=font)


    print(f"Найдено {len(result.boxes)} объектов с уверенностью > {conf}.")

    # page_with_boxes = result.plot()
    if draw_graphs:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(20, 7))

        axs[0].imshow(orig_img)
        axs[0].set_title('Исходное изображение')
        axs[0].axis('off')
        
        axs[1].imshow(image_with_boxes)
        axs[1].set_title('Результат YOLO 1')
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()
    
    return finded_images, image_with_boxes


if __name__ == '__main__':
    # путь до модели
    best_model_path = 'weights/last.pt'
    
    # путь до изображений, которые нужно сегментировать на слова
    example_images = 'examples'
    # Путь до папки куда будут сохраняться изображения с рамками
    images_output = 'detect outputs/run10_1280px_augmented4'
    # Папка в которые будут сохраняться изображения слов
    words_output_dir = ''

    for image in os.listdir(example_images):
        image_path = os.path.join(example_images, image)
        image_w_boxes = detect(best_model_path, image_path, draw_graphs=False,output_folder=words_output_dir)[1]

        os.makedirs(images_output, exist_ok=True)
        image_w_boxes.save(os.path.join(images_output, f'{image}'), 'png')