import pytesseract
import cv2
import re
from tkinter import Tk, filedialog
import json

root = Tk()
root.withdraw()

#file_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("Image files", "*.jpg;*.png")])
file_path = "resource\img1.jpg"


image = cv2.imread(file_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

thresh, im_bw = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

im_bw = cv2.resize(im_bw, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

try:
    custom_config = r'--oem 3 --psm 1'
    text = pytesseract.image_to_string(im_bw, lang='kor+eng')

    lines = text.split('\n')

    user_name = ""
    school_name = ""
    major_name = ""
    student_id = ""

    # 일치시킬 패턴 정의
    user_name_pattern = re.compile(r'^.{3}$')
    school_name_pattern = re.compile(r'대학교')
    major_name_pattern = re.compile(r'과')
    student_id_pattern = re.compile(r'\d{8}')

    # 패턴에 기반하여 정보 추출
    for line in lines:
        if user_name_pattern.match(line):
            user_name = line
        elif school_name_pattern.search(line):
            school_name = line
        elif major_name_pattern.search(line):
            major_name = line
        elif student_id_pattern.match(line):
            student_id = line

    # 결과 딕셔너리 생성
    result = {
        "user_name": user_name,
        "school_name": school_name,
        "major_name": major_name,
        "student_id": student_id
    }

    with open('pytesseract_algorithm.json', 'w', encoding='utf-8') as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)


except Exception as e:
    print(f"오류 발생: {e}")
