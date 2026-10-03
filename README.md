# 다국어 실행 추적 튜터

C, C++, Java, Python 코드를 분석해 **최대 7개 핵심 3지선다 문항**을 만듭니다. Google 로그인 후 각자의 OpenAI API 키를 한 번 등록하면 PC와 휴대전화에서 문제를 이어 풀고 복습할 수 있습니다. 사용자의 코드를 실행하지 않고 모델이 분석합니다.

## 필요한 서비스 설정

1. **Supabase:** [프로젝트 목록](https://supabase.com/dashboard/projects)에서 이 앱 전용 프로젝트 `execution-trace-tutor`를 만들고 [schema.sql](schema.sql)을 SQL Editor에서 한 번 실행합니다. **Project Settings → API Keys**에서 서버용 `sb_secret_...` 키, **Project Settings → Data API**에서 프로젝트 URL을 확인합니다. 데이터베이스는 서버에서만 접근하며, SQL은 익명·일반 역할의 테이블 접근을 차단합니다.
2. **Google 로그인:** [앱 전용 Google Cloud 프로젝트](https://console.cloud.google.com/auth/overview?project=execution-trace-tutor)에서 인증 플랫폼을 설정합니다. 앱 이름은 `실행 추적 튜터`, 대상은 `외부`입니다. **클라이언트 → 클라이언트 만들기 → 웹 애플리케이션**에서 승인된 리디렉션 URI `https://<배포된-앱-주소>/oauth2callback`을 추가합니다. 로컬 실행도 필요하면 `http://localhost:8501/oauth2callback`을 추가합니다. **대상 → 앱 게시**로 게시 상태를 확인합니다. 클라이언트 ID와 비밀값을 준비합니다.
3. **암호화 키:** 아래 명령으로 Fernet 키를 한 번 생성합니다. 이 값을 바꾸면 기존 저장 키를 읽을 수 없으므로 안전하게 보관합니다.

   ```bash
   .venv/bin/python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
   ```

## 로컬 실행

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
.venv/bin/python -m streamlit run app.py
```

`.streamlit/secrets.toml`에서 Supabase URL·서버 키, 고정 암호화 키, Google 클라이언트 ID·비밀값과 로컬 `redirect_uri`를 채웁니다. 개인 OpenAI API 키는 로그인 후 앱 사이드바에서 등록합니다. 로그인·저장소 설정 전에는 학습 기능이 열리지 않습니다. 비밀값 파일은 Git에서 제외됩니다.

## Streamlit Community Cloud 배포

현재 GitHub 저장소는 [BettorBoom/execution-trace-tutor](https://github.com/BettorBoom/execution-trace-tutor)입니다. 저장소 소유자가 바뀌었으므로, 예전 `theBettor` 저장소에 연결된 [기존 Streamlit 앱](https://execution-trace-tutor-isldmuvfxksbizyhpjcihb.streamlit.app/)은 새 코드를 자동으로 배포하지 않을 수 있습니다. [Community Cloud 대시보드](https://share.streamlit.io)에서 새 앱을 만들고 `BettorBoom/execution-trace-tutor`, `main`, `app.py`를 지정하세요. 새 앱이 정상 동작하는 것을 확인한 뒤 기존 앱을 정리하거나, 가능하다면 기존 주소를 새 앱에 배정하세요.

새 앱의 **⋮ → Settings → Secrets**에 [.streamlit/secrets.toml.example](.streamlit/secrets.toml.example)의 항목을 입력합니다. `redirect_uri`에는 **새 앱의 실제 주소** 뒤에 `/oauth2callback`을 붙인 값을 넣고, Google Cloud의 승인된 리디렉션 URI에도 똑같은 값을 추가합니다. `cookie_secret`은 `python3 -c 'import secrets; print(secrets.token_urlsafe(48))'`로 생성할 수 있습니다. Supabase URL·서버 키, **기존과 동일한** 암호화 키, Google 클라이언트 값이 모두 필요합니다. 특히 암호화 키를 바꾸면 기존에 저장된 개인 API 키를 읽을 수 없습니다. 문의 메일은 기본값 `be0128st@gmail.com`이며 Secrets의 `CONTACT_EMAIL`로 변경할 수 있습니다.

배포 후 앱에서 Google 로그인을 누르고, 사이드바에 자신의 OpenAI API 키를 등록한 다음 예제 코드를 생성·풀이·새로고침해 보관함 복원을 확인합니다. **실제 API 키·데이터베이스 비밀번호·클라이언트 비밀값은 GitHub나 이슈에 게시하지 않습니다.**

Streamlit 인증 앱은 Community Cloud의 비공개 앱 한도에 포함됩니다. Google 로그인은 Streamlit이 처리하고, 사용자의 API 키는 암호화된 형태로 Supabase에 저장됩니다. Supabase 서버용 키와 암호화 키는 공개 저장소에 넣지 마세요.

## 검증

```bash
.venv/bin/python -m unittest test_app.py test_storage.py
.venv/bin/python -m pip check
```

모의 테스트에는 외부 계정이 필요하지 않습니다. 실제 Google 로그인·Supabase 접근·OpenAI 생성은 각 서비스 설정 후 별도로 확인해야 합니다. 생성 실패 시 이전 문제는 유지되고, DB 저장 실패 후에는 생성된 결과를 API 재호출 없이 저장 재시도할 수 있습니다.
