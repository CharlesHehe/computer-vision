import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# matplotlib style 지정
# 원하는 스타일로 지정하면 됨
import warnings

warnings.filterwarnings('ignore')

# seaborn 그래프 안의 글자 키우는 tip
sns.set(font_scale=2)

data = pd.read_csv('train.csv')

data.isnull().sum()

f, ax = plt.subplots(1, 2, figsize=(18, 8))
# 1, 2 는 각각 행과 열의 수
# 여기까지하면 쉽게 말해 '도화지'가 생긴다고 보면 됨.

data['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Sex', data=data, ax=ax[1])
ax[1].set_title('Survived')
plt.show()



temp_col = 'Pclvalid'
f, ax = plt.subplots(1, 2, figsize=(18,8))
data['Pclass'].value_counts().plot.pie(explode=[0.1,0.1, 0.1],shadow=True, autopct='%1.1f%%', ax=ax[0])
# explode : pie들이 서로 떨어진 정도
# autopct의 '%1.1f%%'에서 . 뒤의 1은 소수점을 몇 자리까지 나타내는지 알려줌, 앞의 숫자는 아무 숫자나 상관없는듯(내가 다른 숫자로 돌려본 결과에 따르면), 'f랑 %는' 문법인듯
# shadow는 말그대로 그림자, 입체감을 주는 요소


# ax[0] : 두 개의 plot 중 왼쪽 거 즉 첫번째 plot, ax[1] : 두번째 plot
ax[0].set_title('Pclass')
ax[0].set_ylabel('') # ax[0].set_title('Survived') 만 했을 때는 title이 y_label위치에 가있는데 그 위치를 상단 가운데로 조정하기 위한 코드, 즉 y_label 삭제


sns.countplot('Pclass', data=data, ax=ax[1])
# 특정 column의 count를 그려주고 싶을 때 seaborn의 countplot 사용
# 구현하고 싶은 column을 ''안에 넣어주면 됨


ax[1].set_title('Pclass')
plt.show()


data
