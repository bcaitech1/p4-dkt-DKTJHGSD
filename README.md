# pstage_04_dkt

## 브랜치 생성
```
git branch taehwan 
```

<br/>

## 브랜치 이동
```
git checkout taehwan
```

<br/>

## add, commit, push
```
git add .
git commit -m "commit message"
git push origin taehwan
```
<br/>

## commit message rule
```
# 제목은 최대 50글자까지 작성: ex) <feat> Add Login

# 본문은 아래에 작성

# 꼬릿말은 아래에 작성: ex) Github issue #1


# --- COMMIT END ---
#   <타입> 리스트
#   feat    : 새로운 기능추가
#   fix     : 버그 혹은 기능 수정
#   refactor: 리팩토링
#   style   : 코드 스타일 변경
#   docs    : 문서 (추가, 수정, 삭제)
#   test    : 테스트 (테스트 코드 추가, 수정, 삭제)
#   chore   : 기타 변경사항 (빌드 스크립트 수정 등)
# ------------------
#   제목
#   1. 첫글자는 대문자로
#   2. 명령문으로 작성
#   3. 끝에 마침표(./,) 금지
#     4. 제목과 본문은 한줄 띄워 분리
#
#   본문
#   1. - 로 시작
#   2. What, Why, How 중 1개 이상 설명
# ------------------
```

<br/>

## command with your save model file name
```
python train.py -save_name my_model ==> save_dir에 my_model.pt로 저장됨.
default save name = default.pt
```
