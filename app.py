import os
import cv2
import face_recognition
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Папка для хранения изображений
UPLOAD_FOLDER = 'uploads'
KNOWN_FACES_DIR = 'img'

# Разрешенные расширения файлов
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Настроим Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Функция для проверки разрешенных расширений
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Функция для загрузки известных лиц
def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []
    
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)
            if face_encoding:  # Если лицо найдено
                known_face_encodings.append(face_encoding[0])
                known_face_names.append(filename.split('.')[0])  # Имя файла без расширения
    
    return known_face_encodings, known_face_names

# Загружаем известные лица
known_face_encodings, known_face_names = load_known_faces(KNOWN_FACES_DIR)

# Папка для сохранения загруженных изображений
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400

    image_file = request.files['image']
    
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if image_file and allowed_file(image_file.filename):
        filename = secure_filename(image_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Сохраняем изображение
        image_file.save(filepath)
        print(f"File saved to: {filepath}")
        
        # Обрабатываем изображение
        result = process_image(filepath)
        
        # Удаляем изображение после обработки
        os.remove(filepath)
        print(f"File deleted: {filepath}")
        
        return jsonify(result), 200
    else:
        return jsonify({"error": "Invalid file format"}), 400

def process_image(image_path):
    # Загружаем изображение с помощью OpenCV
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Находим все лица на изображении
    face_locations = face_recognition.face_locations(rgb_image)

    if not face_locations:
        return {"message": "No faces detected in the image."}
    
    # Получаем признаки лиц с изображения
    unknown_face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    found_faces = []
    
    for unknown_face_encoding in unknown_face_encodings:
        # Сравниваем с известными лицами
        matches = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)
        
        if True in matches:
            first_match_index = matches.index(True)
            matched_face_name = known_face_names[first_match_index]
            found_faces.append(f"Знакомое лицо найдено: {matched_face_name}")
        else:
            found_faces.append("Это лицо не знакомо!")
    
    if found_faces:
        return {"message": ", ".join(found_faces)}
    else:
        return {"message": "No recognizable faces found."}

if __name__ == '__main__':
    # Запуск Flask сервера
    print("Starting Flask server...")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

