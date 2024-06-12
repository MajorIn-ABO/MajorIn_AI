from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
import cv2
import numpy as np
import re
import json

# FastAPI 인스턴스 생성
app = FastAPI(title="OCR Service API")

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    #특정 도메인만 허용하게 하려면 프론트엔드 배포 사이트 주소만 넣는다
    #allow_origins=["http://localhost:3000"]
    allow_origins=["*"], # 허용할 도메인 목록 # 모든 도메인에서의 요청 허용
    allow_credentials=True, # 자격 증명(쿠키, HTTP 인증)을 포함한 요청을 허용할지 여부 #허용
    allow_methods=["*"], # 허용할 HTTP 메서드 목록. # 모든 메서드를 허용
    allow_headers=["*"], # 허용할 HTTP 헤더 목록. # 모든 헤더를 허용
)

import logging
# uvicorn, paddleocr 로거 설정
for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access", "paddleocr", "ppocr"]:
    logger = logging.getLogger(logger_name)
    logger.handlers = []   # 모든 기존 핸들러 제거
    logger.addHandler(logging.NullHandler())  # 로그 이벤트 무시 핸들러 추가
    logger.propagate = False  # 상위 로거로 이벤트 전파 중지


# 얼굴 인식을 위한 OpenCV 모델 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class OCRService:
    def __init__(self):
        # 한국어 OCR을 위한 PaddleOCR 초기화
        self.ocr = PaddleOCR(use_angle_cls=False, lang='korean')

    '''
    def detect_edits(self, img_array):
        # 엣지 검출을 위해 Canny 엣지 디텍터 사용
        edges = cv2.Canny(img_array, 100, 200)
        # 엣지 픽셀의 비율 계산
        edge_ratio = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        # 과도한 편집이 감지된 경우 True 반환
        return edge_ratio > 0.05  #!! 임계값 조정 가능
    '''
    def preprocess_image(self, img_array):
        # 이미지 이진화 및 전처리
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        denoised = cv2.fastNlMeansDenoising(binary, h=10)
        return denoised
    
    #얼굴 및 IC 칩 감지
    def detect_face_and_ic_chip(self, img_array):
        # 얼굴 감지
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # IC 칩 감지 (예시로 이미지를 오른쪽 하단에서 추출)
        height, width, _ = img_array.shape
        ic_chip_area = img_array[int(height * 0.8):, int(width * 0.8):]
        chip_gray = cv2.cvtColor(ic_chip_area, cv2.COLOR_BGR2GRAY)
        _, chip_binary = cv2.threshold(chip_gray, 128, 255, cv2.THRESH_BINARY)

        # 간단한 휴리스틱: IC 칩의 특징 패턴이 있음
        ic_chip_detected = np.mean(chip_binary) < 200

        return len(faces) > 0, ic_chip_detected

    #이미지의 OCR 실행 및 학생증 정보 추출
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
        major_name_pattern = re.compile(r'.{3,}(과|학부|공학|계열).*')
        student_id_pattern = re.compile(r'\d{7,10}')
        exclusion_pattern = re.compile(r'(학생증|신분증|모바일|이용증|학생명|재발급|썸체크|학부생|주카드|우리|신한|성명|이름|학생)')

        # 초기 값 설정
        user_name, school_name, major_name, student_id = "", "", "", None

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
                # student_id를 정수로 변환하여 저장
                student_id = int(line)  # 숫자로 변환
        '''        
        # 포토샵 감지
        if self.detect_edits(img_array):
            # 포토샵으로 인한 결과
            # return json.dumps({"is_student_id_card": False, "reason": "Image possibly edited"}, ensure_ascii=False, indent=4)
            result_dict = {
                "is_student_id_card" : False
            }
        '''

        if all([user_name, school_name, major_name, student_id]):
        # 결과 딕셔너리 생성
            result_dict = {
                "is_student_id_card" : True,
                "user_name": user_name,
                "school_name": school_name,
                "major_name": major_name,
                "student_id": student_id
            }
        else:
            # 결과 딕셔너리 생성
            result_dict = {
                "is_student_id_card" : False
            }
            #raise HTTPException(status_code=400, detail="Invalid student ID detected.")
        
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
