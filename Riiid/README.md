Riiid 데이터에서 `train.csv` 와 `questions.csv` 파일을 사용해서 우리 데이터와 유사한 파일을 구축했습니다. 데이터 구축 방식은 다음과 같습니다.

- `userID`:  `train.user_id` 사용
- `assessmentItemID`: 'A' + `questions.part` + 0으로 padding된 `train.task_container_id`  + `questions.question_id` (우리 데이터와 비슷하게 10자리로 맞췄습니다.)
- `testId`: 'A' + 0 + `questions.part` + 0 + 00 + 0으로 padding된 `train.task_container_id`  (우리 데이터와 비슷하게 10자리로 맞췄습니다.)
- `answerCode`: `questions.answered_correctly`
- `Timestamp`: `questions.timestamp` 를 사용했는데, Riiid 데이터에서는 첫 번째 interaction을 기준으로 상대적인 시간의 경과만 나와있어서 우리 데이터와 형식만 맞추어, 1970년 1월 1일 0시부터 데이터가 시작합니다. 그래서 주, 요일, 시간 등의 추가적인 feature 생성이 이 데이터에서는 무의미합니다.
- `KnowledgeTag`: `questions.tags` 에서 첫 번째 tag 추출했습니다. 원래는 랜덤하게 추출할까 했는데 데이터가 너무 커서 계산 시간이 오래걸리다보니... 그냥 첫 번째 데이터 추출하는 걸로 변경했습니다.