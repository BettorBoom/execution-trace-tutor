# 다국어 실행 추적 튜터

C, C++, Java, Python 코드를 3지선다 실행 추적 문제로 학습하는 Streamlit 앱입니다. 완료한 문항은 학습 화면의 **지난 단계 다시 보기**에서 질문·정답·해설을 확인할 수 있습니다. 학습 기록은 현재 Streamlit 세션에만 남습니다.

## 로컬 실행

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

사이드바에 OpenAI API 키를 입력하거나, `.streamlit/secrets.toml`에 `OPENAI_API_KEY = "본인 키"`를 설정하세요. 실제 비밀 키 파일은 Git에서 제외됩니다. `.streamlit/secrets.toml.example`은 형식만 보여줍니다.

## Streamlit Community Cloud 배포

1. GitHub에서 빈 `execution-trace-tutor` 저장소를 만듭니다. 이 폴더 자체가 독립 Git 저장소이므로 다른 Playground 파일은 포함하지 않습니다.
2. 이 폴더에서 아래 명령을 실행해 GitHub 원격 저장소에 현재 `main` 브랜치를 올립니다. `<사용자명>`은 본인의 GitHub 이름으로 바꿉니다.

   ```bash
   git remote add origin https://github.com/<사용자명>/execution-trace-tutor.git
   git push -u origin main
   ```

3. [Streamlit Community Cloud](https://share.streamlit.io/)에서 **Create app**을 누르고 방금 올린 저장소, `main` 브랜치, 진입 파일 `app.py`를 선택합니다. `requirements.txt`는 `app.py` 옆에 있어 자동으로 설치됩니다.
4. 본인의 키를 서버에서 쓰려면 **Advanced settings → Secrets**에 아래 내용을 입력합니다. 키를 GitHub 코드나 예시 파일에 넣지 마세요.

   ```toml
   OPENAI_API_KEY = "본인 OpenAI API 키"
   ```

Cloud의 기본 Python 3.12를 선택하면 됩니다. 공개 앱에 서버 공용 키를 설정하면 방문자의 API 호출에도 그 키가 사용됩니다. 개인용이라면 앱 접근 범위를 확인하거나, Secrets를 비워 두고 사이드바에 본인 키를 입력하세요.

## 검증

```bash
.venv/bin/python -m unittest test_app.py
```
