from paddleocr import PaddleOCR
import cv2

def preprocess_image(img_path):
    # 이미지 로드
    img = cv2.imread(img_path)
    # 이미지 이진화
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 이미지 잡음 제거
    denoised = cv2.fastNlMeansDenoising(binary, h=10)
    return denoised

img_path = "resource\img1.jpg"

# 이미지 전처리
preprocessed_img = preprocess_image(img_path)

ocr = PaddleOCR(use_angle_cls=True, lang='korean')
result = ocr.ocr(preprocessed_img, cls=True)

# 결과 처리
extracted_texts = []
for line in result:
    for word in line:
        extracted_texts.append(word[1][0])  # 텍스트 정보만 추출
    
print(extracted_texts)