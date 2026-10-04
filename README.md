# 다국어 실행 추적 튜터

C, C++, Java, Python 코드에서 **최대 7개 핵심 3지선다 문항**을 만듭니다. Google 로그인 후 각자의 OpenAI API 키를 등록하면 PC와 휴대전화에서 이어 풀고 복습할 수 있습니다. OpenAI가 확인할 행·표현식을 제안하고, 앱은 중복을 제거하고 단순 출력 변수의 마지막 대입을 보강합니다. 정답과 숫자 설명은 [Modal Sandbox](https://modal.com/docs/guide/sandboxes)에서 실제 코드를 실행해 관측한 값으로 만듭니다. 제출 코드는 OpenAI와 Modal로 전송됩니다.

학습 형식 v4는 원본을 두 번 실행해 결과가 일정한지 확인하고, 값 질문은 읽기 전용 식을 대입문 전후에 관측합니다. 관측 코드를 넣은 실행의 출력이 원본 출력과 다르면 해당 값 질문을 버립니다. 제어문 본문을 보존하고 재귀 호출별 전후 값을 짝지으며, 한 관측 지점이 실패해도 다른 유효한 지점은 따로 확인합니다. 모델이 만든 숫자·정답·오답은 사용하지 않습니다. 정답이나 패스 뒤에는 현재 행에서 **맞았습니다! / 패스했습니다**, 관측값과 해설을 보여주고 **다음 문제**를 눌러 이동합니다. v1·v2·v3 보관 기록은 원본과 복습용으로 열 수 있지만 검증 전 정답의 채점은 중단합니다. 새 문제 생성은 1회 OpenAI 호출과 격리 실행이 필요합니다.

## 필요한 서비스 설정

1. **Supabase:** [프로젝트 목록](https://supabase.com/dashboard/projects)에서 이 앱 전용 프로젝트 `execution-trace-tutor`를 만들고 [schema.sql](schema.sql)을 SQL Editor에서 한 번 실행합니다. **Project Settings → API Keys**에서 서버용 `sb_secret_...` 키, **Project Settings → Data API**에서 프로젝트 URL을 확인합니다. 데이터베이스는 서버에서만 접근하며, SQL은 익명·일반 역할의 테이블 접근을 차단합니다.
2. **Google 로그인:** [앱 전용 Google Cloud 프로젝트](https://console.cloud.google.com/auth/overview?project=execution-trace-tutor)에서 인증 플랫폼을 설정합니다. 앱 이름은 `실행 추적 튜터`, 대상은 `외부`입니다. **클라이언트 → 클라이언트 만들기 → 웹 애플리케이션**에서 승인된 리디렉션 URI `https://<배포된-앱-주소>/oauth2callback`을 추가합니다. 로컬 실행도 필요하면 `http://localhost:8501/oauth2callback`을 추가합니다. **대상 → 앱 게시**로 게시 상태를 확인합니다. 클라이언트 ID와 비밀값을 준비합니다.
3. **암호화 키:** 아래 명령으로 Fernet 키를 한 번 생성합니다. 이 값을 바꾸면 기존 저장 키를 읽을 수 없으므로 안전하게 보관합니다.

   ```bash
   .venv/bin/python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
   ```
4. **격리 실행:** [Modal 가입 및 API 토큰 설정](https://modal.com/docs/guide/sandboxes)을 마친 뒤 Modal 대시보드의 API 토큰 ID와 비밀값을 `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`으로 서버 Secrets에 넣습니다. 이 토큰은 앱 운영자의 것이며 이용자의 OpenAI 키와 별도입니다. 매 생성마다 요청 전용 샌드박스를 만들고 종료하므로 [Modal 실행 요금](https://modal.com/docs/guide/sandbox-resources)이 발생할 수 있습니다. 샌드박스에는 API 키를 전달하지 않고 외부 네트워크를 차단합니다. 첫 실행은 C/C++·Java 컴파일러 이미지 구축 때문에 느릴 수 있습니다.

## 로컬 실행

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
.venv/bin/python -m streamlit run app.py
```

`.streamlit/secrets.toml`에서 Supabase URL·서버 키, 고정 암호화 키, Google 클라이언트 ID·비밀값, Modal 토큰과 로컬 `redirect_uri`를 채웁니다. 개인 OpenAI API 키는 로그인 후 앱 사이드바에서 등록합니다. Modal 토큰이 없으면 새 문제 생성 버튼이 비활성화됩니다. 비밀값 파일은 Git에서 제외됩니다.

## Streamlit Community Cloud 배포

현재 GitHub 저장소는 [BettorBoom/execution-trace-tutor](https://github.com/BettorBoom/execution-trace-tutor)입니다. GitHub 계정명을 `theBettor`에서 `BettorBoom`으로 바꾼 뒤 [기존 Streamlit 앱](https://execution-trace-tutor-isldmuvfxksbizyhpjcihb.streamlit.app/)의 관리 화면에는 예전 경로가 표시되지만, **Manage app → ⋮ → Reboot app**으로 재부팅하면 GitHub 리디렉션을 통해 최신 `main` 코드를 다시 가져오는 것을 확인했습니다. 이 방식은 기존 주소·Secrets·Google 로그인 콜백을 유지합니다. 이름 변경 후 자동 배포 웹훅은 검증되지 않았으므로, 코드를 올린 뒤 화면의 `앱 버전 4.4 · 실행 검증`과 새 기능을 확인하고 갱신되지 않았다면 재부팅하세요.

나중에 새 GitHub 경로로 Cloud 앱을 다시 만들려면 [Community Cloud 대시보드](https://share.streamlit.io)에서 `BettorBoom/execution-trace-tutor`, `main`, `app.py`를 지정합니다. 이 경우 새 주소에 맞춰 Secrets와 Google 로그인 콜백을 다시 설정해야 합니다.

새 앱으로 옮길 때만 **⋮ → Settings → Secrets**에 [.streamlit/secrets.toml.example](.streamlit/secrets.toml.example)의 항목을 입력합니다. `redirect_uri`에는 **새 앱의 실제 주소** 뒤에 `/oauth2callback`을 붙인 값을 넣고, Google Cloud의 승인된 리디렉션 URI에도 똑같은 값을 추가합니다. `cookie_secret`은 `python3 -c 'import secrets; print(secrets.token_urlsafe(48))'`로 생성할 수 있습니다. Supabase URL·서버 키, **기존과 동일한** 암호화 키, Google 클라이언트 값이 모두 필요합니다. 특히 암호화 키를 바꾸면 기존에 저장된 개인 API 키를 읽을 수 없습니다. 문의 메일은 기본값 `be0128st@gmail.com`이며 Secrets의 `CONTACT_EMAIL`로 변경할 수 있습니다.

배포 후 화면 상단의 `앱 버전 4.4 · 실행 검증` 표시를 확인합니다. Google 로그인과 개인 OpenAI 키 등록 후, C 포인터 예제로 4행의 `4→1`, 13행의 `6→1`, 최종 출력 `1`이 표시되는지 확인합니다. **실제 API 키·Modal 토큰·데이터베이스 비밀번호·클라이언트 비밀값은 GitHub나 이슈에 게시하지 않습니다.**

Streamlit 인증 앱은 Community Cloud의 비공개 앱 한도에 포함됩니다. Google 로그인은 Streamlit이 처리하고, 사용자의 API 키는 암호화된 형태로 Supabase에 저장됩니다. Supabase 서버용 키와 암호화 키는 공개 저장소에 넣지 마세요.

## 검증

```bash
.venv/bin/python -m unittest test_app.py test_storage.py test_verified_trace.py test_trace_worker_regressions.py
.venv/bin/python -m pip check
```

모의 테스트에는 외부 계정이 필요하지 않습니다. `test_verified_trace.py`는 설치된 로컬 컴파일러가 있으면 네 언어의 안전한 고정 예제를 실제 실행합니다. Modal 연결·Cloud 배포와 실제 OpenAI 생성은 운영자 Modal 토큰 설정 후 별도로 확인해야 합니다. 현재 값 질문은 **단일 행 대입문의 정수 표현식**만 지원합니다. 입력이 필요한 코드, 실행 오류, 긴 출력, 감지된 불안정한 실행은 출제하지 않습니다. C/C++의 정의되지 않은 동작은 감지 가능한 범위에서 중단하며 모든 경우를 판별할 수는 없습니다. 생성 실패 시 이전 문제는 유지되고, DB 저장 실패 후에는 생성된 결과를 API 재호출 없이 저장 재시도할 수 있습니다.
