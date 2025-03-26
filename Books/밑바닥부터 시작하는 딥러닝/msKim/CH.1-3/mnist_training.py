import sys, os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist
from PIL import Image 

''' 
 sys.path 
: 파이썬이 모듈을 찾는 경로들의 리스트로, import할 때 이 경로에 있는 파일을 찾을 수 있음음
 os.pardir
: 현재 디렉터리(.)의 부모 디렉터리(..)를 의미함
 sys.path.append(os.pardir)
: 부모 디렉터리를 파이썬 모듈 검색 경로에 추가함
-> 현재 폴더의 부모 디렉터리에 있는 모듈을 import하고 싶을 때 사용함
 
 Python Imaging Library(PIL)
: 이미지를 다루는 파이썬 라이브러리로, 이미지 파일을 열고, 수정하고, 저장하는 기능 제공함

'''

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) # fromarray() : Numpy 배열을 PIL.Image로 변환함
    pil_img.show() 
    # OS 기본 이미지 뷰어에서 이미지 출력함 -> PIL은 이미지를 python 창이 아니라, os 이미지 창에 띄우게 해줌 
    
# \ : 긴 코드 줄 바꿈하기 위해 사용함 콘솔창에서는 일렬로 출력됨
''' load_mnist()는 읽은 Mnist 데이터를 
(훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블) 형식으로 반환함
인수로는 normalize, flatten, one_hot_label 세 가지를 설정할 수 있음
'''
(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False, one_hot_label=False)

# Mnist 데이터셋에서 첫 번째 훈련 데이터(이미지,레이블)를 출력함
img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28) # 원하는 형상으로 넘파이 배열을 재형성성
print(img.shape) # 이미지 차원 출력

img_show(img)

# PIL vs Matplotlib 어떤 방식으로 이미지 출력할지 생각해보기
# 언제 one_hot_label = True를 써야 할지 생각해보기
