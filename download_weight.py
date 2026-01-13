import os
import gdown

def download_weights():
    file_id = '1woLCmXIQG18JBHPXusvELASPMDM49pCM' 
    output_folder = 'weights'
    output_file = 'best.pt'
    destination = os.path.join(output_folder, output_file)

    if os.path.exists(destination):
        print(f"Файл {destination} уже существует")
        return
    os.makedirs(output_folder, exist_ok=True)

    print(f"Скачивание весов модели YOLO в {destination}")
    
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, destination, quiet=False)
    print("\nВеса скачаны.")

if __name__ == "__main__":
    download_weights()