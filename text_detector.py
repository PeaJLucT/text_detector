from PIL import Image, ImageDraw, ImageOps
import os
from ultralytics import YOLO
import numpy as np

def sort_boxes(boxes):
    """
    Сортирует рамки в порядке чтения: сверху-вниз, слева-направо
    """
    if not boxes:
        return []
    # определяем среднюю высоту рамок, чтобы задать допуск для одной строки
    avg_height = np.mean([b.xyxy[0][3] - b.xyxy[0][1] for b in boxes])
    # print(f'Средний допуск строки - {avg_height}')
    
    # Сортируем рамки по строкам
    # b.xyxy[0][1] // (avg_height * 0.7) - это строка
    sorted_boxes = sorted(boxes, key=lambda b: (b.xyxy[0][1] // (avg_height * 0.7), b.xyxy[0][0]))
    
    return sorted_boxes


def detect(model_path: str, image_path: str, draw_graphs = False, conf = 0.5, output_folder = 'output') -> list:
    '''
    Возвращает список обнаруженных и вырезанных слов с изображения

    Args:
        model_path (str): Путь до лучшей модели
        image_path (str): Путь до изображения *.jpg | *.png
        draw_graphs (bool): True - если нужно вывести в output изображение оригинальное и с маской найденных слов
        conf (float): Уверенность модели для записи маски как правильной
        output_folder (str): Путь до папки, в которую будут сохранены все вырезанные найденные слова с изображения
    Returns:
        list: [Список найденных изображений, оригинальное изображение с маской детекции]
    '''
    
    try:
        model = YOLO(model_path, task='detect')
        print(f"Лучшая модель успешно загружена из {model_path}")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return []

    try:
        orig_img = Image.open(image_path).convert('RGB')
        orig_img = ImageOps.exif_transpose(orig_img)
        print(f'Изображение успешно загружено из {image_path}')
    except Exception as e:
        print(f"Ошибка при открытии изображения: {e}")
        return []
    
    # Детектим слова
    results = model(orig_img, conf=conf)
    result = results[0]

    sorted_boxes = sort_boxes(result.boxes)
    
    image_with_boxes = orig_img.copy()
    draw = ImageDraw.Draw(image_with_boxes)

    # список найденных слов 
    finded_images = list()

    #папка для сохранения изображений слов
    os.makedirs(output_folder, exist_ok=True)

    for i, box in enumerate(sorted_boxes):
        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
        confid = float(box.conf[0])
        
        # Вырезаем слово
        crop_image = orig_img.crop((x1, y1, x2, y2))
        crop_image.save(os.path.join(output_folder, f'{i}.jpg'), quality = 100)
        finded_images.append(crop_image)

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        label = f"{confid:.2f}"
        draw.text((x1, y1 - 15), label, fill="red")


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
    
    return [finded_images, image_with_boxes]


if __name__ == '__main__':
    # путь до модели
    best_model_path = 'weights/best.pt'
    
    # путь до изображения, которое нужно сегментировать на слова
    # image_path = 'INFO/examples/6.jpg'
    images = 'INFO/examples'
    images_output = 'detect outputs/75 + 75/'
    for image in os.listdir(images):
        image_path = os.path.join(images, image)
        image_w_boxes = detect(best_model_path, image_path, draw_graphs=False)[1]
        image_w_boxes.save(os.path.join(images_output, f'{image}'), 'png')