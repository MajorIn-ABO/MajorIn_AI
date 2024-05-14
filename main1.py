from fastapi import FastAPI, UploadFile, File, HTTPException
from paddleocr import PaddleOCR
import cv2
import numpy as np
import re
import json

# FastAPI 인스턴스 생성
app = FastAPI(title="OCR Service API")

class OCRService:
    def __init__(self):
        # 한국어 OCR을 위한 PaddleOCR 초기화
        self.ocr = PaddleOCR(use_angle_cls=False, lang='korean')

    def preprocess_image(self, img_array):
        # 이미지 이진화 및 전처리
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        denoised = cv2.fastNlMeansDenoising(binary, h=10)
        return denoised

    def perform_ocr_and_extract(self, img_array):
        # 이미지 전처리
        preprocessed_img = self.preprocess_image(img_array)

        # 텍스트 인식
        result = self.ocr.ocr(preprocessed_img, cls=True)

        # 인식된 텍스트 추출 및 결합
        extracted_texts = [word[1][0] for line in result for word in line]
        text = "\n".join(extracted_texts)

        # 패턴 정의
        user_name_pattern = re.compile(r'^[가-힣]{2,3}$')
        school_name_pattern = re.compile(r'(.*대학교)')
        major_name_pattern = re.compile(r'.*(학과|학부|공학|계열).*')
        student_id_pattern = re.compile(r'\d{7,10}')
        exclusion_pattern = re.compile(r'(학생증|신분증|모바일|이용증|학생명|재발급|썸체크|학부생|우리|신한|성명|이름|학생)')

        # 초기 값 설정
        user_name, school_name, major_name, student_id = "", "", "", ""

        # 패턴을 기반으로 정보 추출
        lines = text.split('\n')
        for line in lines:
            if user_name_pattern.match(line) and not exclusion_pattern.search(line) and not user_name:
                user_name = line
            if school_name_pattern.match(line) and not school_name:
                school_name = school_name_pattern.match(line).group(1).split('대학교')[0] + '대학교'
            if major_name_pattern.match(line) and not exclusion_pattern.search(line) and not major_name:
                major_name = line
            if student_id_pattern.match(line) and not student_id:
                student_id = line

        # 결과 딕셔너리 생성
        result_dict = {
            "user_name": user_name,
            "school_name": school_name,
            "major_name": major_name,
            "student_id": student_id
        }

        # JSON 반환
        return json.dumps(result_dict, ensure_ascii=False, indent=4)

@app.get("/")
def read_root():
    return {"message": "Welcome to the OCR API service."}

@app.post("/perform_ocr/")
async def perform_ocr(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # OCR을 통해 추출 및 패턴 매칭 수행
        ocr_service = OCRService()
        json_result = ocr_service.perform_ocr_and_extract(img)
        return json.loads(json_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
