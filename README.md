# Restaurant Bot: Handoffs

OpenAI Agents SDK와 Streamlit으로 만든 레스토랑 고객지원 에이전트입니다.

## 기능

- Triage Agent가 고객 의도를 파악합니다.
- Menu Agent가 메뉴, 재료, 알레르기, 채식 옵션에 답합니다.
- Order Agent가 주문을 받고 확인합니다.
- Reservation Agent가 예약 정보를 수집하고 확인합니다.
- Streamlit UI에 handoff 실행 로그를 표시합니다.

## 실행

```bash
uv run --python 3.12 --with-requirements requirements.txt streamlit run app.py
```

`DEEPSEEK_API_KEY`를 환경변수 또는 Streamlit Secrets에 등록해야 합니다.

## 모델

기본 모델은 비용 효율을 위해 `deepseek-v4-flash`를 사용합니다. 사이드바에서 `deepseek-v4-pro`도 선택할 수 있습니다.
