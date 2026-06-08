from PIL import Image, ImageDraw, ImageOps, ImageFont, ImageEnhance
import PIL
import os
from ultralytics import YOLO
import ultralytics
import numpy as np
import cv2


def sort_boxes(boxes):
    """
    Сортирует рамки в порядке чтения: сверху-вниз, слева-направо
    """
    if not boxes:
        return []
    # определяем среднюю высоту рамок, чтобы задать допуск для одной строки
    boxes_data = np.array([b.data[0].cpu().numpy() for b in boxes])
    heights = boxes_data[:, 3] - boxes_data[:, 1]
    avg_height = np.mean(heights)
    # print(f'Средний допуск строки - {avg_height}')
    
    # Сортируем рамки по строкам
    sorted_boxes = sorted(boxes, key=lambda b: (b.xyxy[0][1].item() // (avg_height * 0.7), b.xyxy[0][0].item()))
    
    return sorted_boxes


def filter_nested_boxes_robust(boxes, threshold=0.90):
    """
    Удаляет вложенные боксы (меньшие боксы внутри больших) одной модели
    Принимает на вход как YOLO Boxes/Python-список боксов
    Args:
        threshold: процент площади меньшего бокса, который должен быть внутри большего (0.90 = 90%)
    Returns:
        result_boxes (list): [yolo_box1, yolo_box2,...,yolo_boxn]
    """
    list_b = [b for b in boxes] if boxes is not None else []
    # Если боксов нет, возвращаем
    if len(list_b) == 0:
        return list_b

    def get_area(b):
        coords = b.xyxy[0].tolist()
        return max(0, coords[2] - coords[0]) * max(0, coords[3] - coords[1])
    
    sorted_indices = sorted(range(len(list_b)), key=lambda idx: get_area(list_b[idx]), reverse=True)
    keep_flags = [True] * len(list_b)

    for i in range(len(sorted_indices)):
        idx_out = sorted_indices[i]
        if not keep_flags[idx_out]:
            continue
        
        x1_out, y1_out, x2_out, y2_out = list_b[idx_out].xyxy[0].tolist()

        for j in range(i + 1, len(sorted_indices)):
            idx_in = sorted_indices[j]
            if not keep_flags[idx_in]:
                continue
            
            x1_in, y1_in, x2_in, y2_in = list_b[idx_in].xyxy[0].tolist()
            area_in = get_area(list_b[idx_in])

            if area_in == 0:
                keep_flags[idx_in] = False
                continue

            # Находим координаты пересечения
            ix1 = max(x1_out, x1_in)
            iy1 = max(y1_out, y1_in)
            ix2 = min(x2_out, x2_in)
            iy2 = min(y2_out, y2_in)

            # Площадь пересечения
            intersection_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)

            # Какая часть меньшего бокса перекрыта большим боксом
            overlap_ratio = intersection_area / area_in

            # Если меньший бокс находится внутри большего на 'threshold' процентов и более
            if overlap_ratio >= threshold:
                keep_flags[idx_in] = False

    result_boxes = [list_b[idx] for idx in range(len(list_b)) if keep_flags[idx]]
    return result_boxes


def merge_two_model_detections(boxes1, boxes2, priority=1, iou_thresh=0.8):
    """
    Объединяет детекции двух моделей YOLO
    При пересечении приоритет отдается модели, указанной в параметре priority
    
    Args:
        boxes1: Результаты первой модели (YOLO Boxes)
        boxes2: Результаты второй модели (YOLO Boxes)
        priority: 1 для приоритета boxes1, 2 для приоритета boxes2
        iou_thresh: Порог пересечения (IoU) для фильтрации дубликатов
    Returns:
        boxes (list): list[yolo_box1, yolo_box2,...,yolo_boxn]
    """

    list1 = [b for b in boxes1] if boxes1 is not None else []
    list2 = [b for b in boxes2] if boxes2 is not None else []
    # Если боксов нет, возвращаем
    if len(list1) == 0:
        return list2
    elif len(list2) == 0:
        return list1

    # если приоритет 1, то выбираем как основу 1 модель, иначе вторую
    if priority == 1:
        primary_boxes = list1
        secondary_boxes = list2
    else:
        primary_boxes = list2
        secondary_boxes = list1

    keep_secondary = []

    def calculate_iou(box_a, box_b):
        # Получаем координаты [x1, y1, x2, y2]
        coords_a = box_a.xyxy[0].tolist()
        coords_b = box_b.xyxy[0].tolist()

        # Координаты пересечения
        ix1 = max(coords_a[0], coords_b[0])
        iy1 = max(coords_a[1], coords_b[1])
        ix2 = min(coords_a[2], coords_b[2])
        iy2 = min(coords_a[3], coords_b[3])

        # Площадь пересечения
        inter_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter_area == 0.0:
            return 0.0

        area_a = (coords_a[2] - coords_a[0]) * (coords_a[3] - coords_a[1])
        area_b = (coords_b[2] - coords_b[0]) * (coords_b[3] - coords_b[1])
        # Какая часть меньшего бокса перекрыта большим боксом
        overlap_ratio1 = inter_area / area_a
        overlap_ratio2 = inter_area / area_b

        return max(overlap_ratio1, overlap_ratio2)

    for sec_box in secondary_boxes:
        has_overlap = False
        for prim_box in primary_boxes:
            iou = calculate_iou(sec_box, prim_box)
            if iou >= iou_thresh:
                has_overlap = True
                break  
    
        if not has_overlap:
            keep_secondary.append(sec_box)

    # Объединяем приоритетные боксы и оставшиеся второстепенные
    return primary_boxes + keep_secondary

def detect(model_path: str, image_path: str, draw_graphs = False, conf = 0.3, output_folder = '', threshold_value= 0.9, model_path_2 = None) -> tuple[list[PIL.Image.Image], PIL.Image.Image | None, int, list]:
    '''
    Возвращает список обнаруженных и вырезанных слов с изображения

    Args:
        model_path (str): Путь до лучшей модели сегментации
        model_path_2 (str): Путь до второй модели сегментации | None
        image_path (str): Путь до изображения *.jpg | *.png
        draw_graphs (bool): True - если нужно вывести в output изображение оригинальное и с маской найденных слов
        conf (float): Уверенность модели для записи маски как правильной
        output_folder (str): Путь до папки, в которую будут сохранены все вырезанные найденные слова с изображения
    Returns:
        list: [ [Список найденных изображений] , оригинальное изображение с маской детекции, количество найденных слов на изображении, [список отсортированных боксы]]
    '''

    # Загружаем модели
    try:
        model = YOLO(model_path, task='detect')
        print(f"Лучшая модель успешно загружена из {model_path}")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return [], None

    # Если вторая модель подана на вход, то берем ее
    if model_path_2 is not None:
        try:
            model_2 = YOLO(model_path_2, task='detect')
            print(f"Вторая модель успешно загружена из {model_path_2}")
        except Exception as e:
            model_2 = False
            print(f"Ошибка при загрузке второй модели: {e}")
            return [], None
    else:
        model_2 = False

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

    results = model.predict(img_gray_for_model, imgsz=1280, conf=conf, verbose=False)           #1 модель
    result = results[0]
    # 1 сортировка боксов по вхождения друг в друга
    filtered_boxes = filter_nested_boxes_robust(result.boxes, threshold=threshold_value)              #1 модель - фильтрация боксов
    if model_2:
        results2 = model_2.predict(img_gray_for_model, imgsz=1280, conf=conf, verbose=False)    #2 модель
        result2 = results2[0]
        filtered_boxes2 = filter_nested_boxes_robust(result2.boxes, threshold=threshold_value)        #2 модель - фильтрация боксов
        merged_boxes = merge_two_model_detections(filtered_boxes, filtered_boxes2, priority=1, iou_thresh=0.7)
        # 2 сортировка боксов по порядку/строкам
        sorted_boxes = sort_boxes(merged_boxes)
    else:
        sorted_boxes = sort_boxes(filtered_boxes)


    image_with_boxes = orig_img.copy()
    # draw = ImageDraw.Draw(image_with_boxes)
    #
    # try:
    #     font = ImageFont.truetype("arial.ttf", 40)
    # except IOError:
    #     font = ImageFont.load_default()

    # список найденных слов
    finded_images = list()

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    if output_folder:
        current_img_output = os.path.join(output_folder, image_name)
        #папка для сохранения изображений слов
        os.makedirs(current_img_output, exist_ok=True)

    for i, box in enumerate(sorted_boxes):
        c = box.xyxy[0].tolist()
        x1, y1, x2, y2 = c[0], c[1], c[2], c[3]

        confid = float(box.conf[0])
        
        # Вырезаем слово
        crop_image = orig_img.crop((x1, y1, x2, y2))
        if output_folder:
            crop_image.save(os.path.join(current_img_output, f'{i}.jpg'), quality = 100)
        finded_images.append(crop_image)

        # draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
        #
        # label = f"{confid:.2f}"
        # text_bbox = draw.textbbox((x1, y1), label, font=font)
        # draw.rectangle((x1, y1 - (text_bbox[3]-text_bbox[1]), x1 + (text_bbox[2]-text_bbox[0]), y1), fill="red")
        # draw.text((x1, y1 - (text_bbox[3]-text_bbox[1])), label, fill="white", font=font)


    print(f"Найдено {len(sorted_boxes)} объектов с уверенностью > {conf}.")

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
    
    return finded_images, image_with_boxes, len(sorted_boxes), sorted_boxes


if __name__ == '__main__':
    # путь до модели
    # best_model_paths = ['DATASETS FOR TRAIN/TRAIN RESULTS/1280px_100_medium_1/weights/best.pt',
    #                     'DATASETS FOR TRAIN/TRAIN RESULTS/1280px_100_medium_2/weights/best.pt',
    #                     'DATASETS FOR TRAIN/TRAIN RESULTS/1280px_100_medium_3/weights/best.pt',
    #                     'DATASETS FOR TRAIN/TRAIN RESULTS/1280px_100_medium_4/weights/best.pt',
    #                     'DATASETS FOR TRAIN/TRAIN RESULTS/1280px_100_small_1/weights/best.pt',
    #                     'DATASETS FOR TRAIN/TRAIN RESULTS/1280px_100_small_2/weights/best.pt',
    #                     'DATASETS FOR TRAIN/TRAIN RESULTS/1280px_100_small_3/weights/best.pt',
    #                     'DATASETS FOR TRAIN/TRAIN RESULTS/1280px_100_small_4/weights/best.pt',
    #                     'DATASETS FOR TRAIN/TRAIN RESULTS/run10_1280px_augmentedTrain/weights/best.pt'
    #                     ]
    best_model_1 = 'DATASETS FOR TRAIN/TRAIN RESULTS/1280px_100_medium_4/weights/best.pt'
                        
    best_model_2 = 'DATASETS FOR TRAIN/TRAIN RESULTS/1280px_100_medium_1/weights/best.pt'
                        
    # путь до изображений, которые нужно сегментировать на слова
    example_images = "text_detector/museum's  photo examples"
    # Путь до папки куда будут сохраняться изображения с рамками
    images_output = 'detect outputs_3/'
    # Папка в которые будут сохраняться изображения слов
    words_output_dir = ''

    import pandas as pd
    df_results = pd.DataFrame(columns=["model", "image", "number of found"])

    for image in os.listdir(example_images):
        model_train_path = best_model_1.split('/')[2] + best_model_2.split('/')[2]
        image_path = os.path.join(example_images, image)
        image_w_boxes, founded_count, _  = detect(best_model_1, image_path, draw_graphs=False,output_folder=words_output_dir, threshold_value=0.9, model_path_2=best_model_2)[1:]

        output = os.path.join(images_output, model_train_path)
        os.makedirs(output, exist_ok=True)
        image_w_boxes.save(os.path.join(output, f'{image}'), 'png')
        new_data = {
            "model": model_train_path, 
            "image": image, 
            "number of found": founded_count 
        }
        df_results.loc[len(df_results)] = new_data



        # for best_model in best_model_paths:
            # model_train_path = best_model.split('/')[2]
            # image_path = os.path.join(example_images, image)
            # image_w_boxes, founded_count  = detect(best_model, image_path, draw_graphs=False,output_folder=words_output_dir, threshold_value=0.9)[1:]

            # output = os.path.join(images_output, model_train_path)
            # os.makedirs(output, exist_ok=True)
            # image_w_boxes.save(os.path.join(output, f'{image}'), 'png')
            # new_data = {
            #     "model": model_train_path, 
            #     "image": image, 
            #     "number of found": founded_count 
            # }
            # df_results.loc[len(df_results)] = new_data
    
    # сохраняем табилцу с результатами каждой обученной модели и найденных ею слов
    df_results.to_csv("model_results_2.csv", index=False, encoding='utf-8')
    print("Данные успешно сохранены в model_results.csv")