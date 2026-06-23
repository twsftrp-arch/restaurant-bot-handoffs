import os
import time
import uuid
from pathlib import Path

import streamlit as st
from agents import (
    Agent,
    HandoffOutputItem,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunHooks,
    Runner,
    SQLiteSession,
    handoff,
    set_tracing_disabled,
)
from openai import AsyncOpenAI


APP_TITLE = "Restaurant Bot"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
MODEL_OPTIONS = {
    "deepseek-v4-flash": "Flash",
    "deepseek-v4-pro": "Pro",
}
DB_PATH = Path("restaurant_bot_sessions.db")
FALLBACK_ENV_PATH = Path.home() / "Documents" / "movie-agent" / ".env"


MENU_TEXT = """
Signature Menu
- Truffle Mushroom Risotto: arborio rice, mushroom stock, parmesan, truffle oil. Vegetarian. Contains dairy.
- Spicy Seafood Pasta: linguine, shrimp, squid, tomato chili sauce. Contains shellfish and gluten.
- Grilled Chicken Salad: chicken breast, greens, avocado, lemon vinaigrette. Gluten-free.
- Vegan Grain Bowl: quinoa, roasted vegetables, chickpeas, tahini dressing. Vegan. Contains sesame.
- Classic Cheeseburger: beef patty, cheddar, lettuce, tomato, brioche bun. Contains dairy and gluten.
- Chocolate Lava Cake: warm chocolate cake, vanilla ice cream. Contains dairy, eggs, and gluten.

Drinks
- Sparkling Lemonade
- Iced Americano
- House Red Wine
- Zero Sugar Cola
""".strip()


TRIAGE_INSTRUCTIONS = """
You are the Triage Agent for a restaurant customer support bot.

Your job is routing, not answering. Decide what the customer wants and hand off to exactly one specialist agent:
- Menu Agent: menu items, ingredients, allergies, vegetarian/vegan/gluten-free options, recommendations.
- Order Agent: placing, changing, checking, or confirming food orders.
- Reservation Agent: booking, changing, or checking table reservations.

Rules:
- Always use a handoff for restaurant requests.
- If the message mixes intents, route to the most immediate requested task.
- If the customer changes topic, hand off to the new appropriate specialist.
- Do not answer menu, order, or reservation questions yourself.
- Keep any routing text very short.
""".strip()


MENU_INSTRUCTIONS = f"""
You are the Menu Agent.

Answer questions about menu items, ingredients, allergies, dietary restrictions, and recommendations.
Use only this menu data unless the customer asks for general preference guidance:

{MENU_TEXT}

Style:
- Be friendly, concise, and practical.
- If allergies are mentioned, clearly name relevant allergens and advise the guest to confirm with staff for severe allergies.
- If the customer wants to order after discussing the menu, ask them what they would like to order; the next turn can be routed to Order Agent.
""".strip()


ORDER_INSTRUCTIONS = f"""
You are the Order Agent.

Help the customer place or revise an order from this menu:

{MENU_TEXT}

Collect and confirm:
- item names
- quantities
- dine-in or takeout
- customer name if needed
- any allergy or special request

Do not invent payment processing. End with a clear order summary and ask for confirmation if required details are missing.
""".strip()


RESERVATION_INSTRUCTIONS = """
You are the Reservation Agent.

Help the customer book or change a table reservation.
Collect and confirm:
- date
- time
- party size
- customer name
- phone number if they offer it
- seating preference if relevant

Do not claim a real reservation has been saved in an external system. Say you can prepare/confirm the reservation details for staff.
""".strip()


class HandoffLogger(RunHooks):
    def __init__(self, started_at: float) -> None:
        self.started_at = started_at
        self.events: list[dict[str, str]] = []

    def _add(self, message: str) -> None:
        elapsed = time.perf_counter() - self.started_at
        self.events.append({"time": f"{elapsed:.2f}s", "message": message})

    async def on_agent_start(self, context, agent) -> None:
        self._add(f"{agent.name} 실행 시작")

    async def on_handoff(self, context, from_agent, to_agent) -> None:
        self._add(f"{from_agent.name} → {to_agent.name} handoff")

    async def on_agent_end(self, context, agent, output) -> None:
        self._add(f"{agent.name} 응답 완료")


def load_deepseek_api_key() -> str:
    env_key = os.getenv("DEEPSEEK_API_KEY")
    if env_key:
        return env_key

    try:
        secret_key = st.secrets.get("DEEPSEEK_API_KEY")
        if secret_key:
            return str(secret_key)
    except Exception:
        pass

    if FALLBACK_ENV_PATH.exists():
        for line in FALLBACK_ENV_PATH.read_text(encoding="utf-8").splitlines():
            if not line.startswith("DEEPSEEK_API_KEY="):
                continue
            _, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            if value:
                return value

    return ""


def build_model(model_name: str, api_key: str) -> OpenAIChatCompletionsModel:
    client = AsyncOpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    return OpenAIChatCompletionsModel(model=model_name, openai_client=client)


def build_agents(model_name: str, api_key: str) -> Agent:
    model = build_model(model_name, api_key)
    settings = ModelSettings(
        parallel_tool_calls=False,
        max_tokens=900,
        extra_body={"thinking": {"type": "disabled"}},
    )

    menu_agent = Agent(
        name="Menu Agent",
        handoff_description="Specialist for menu, ingredients, allergies, and dietary questions.",
        model=model,
        model_settings=settings,
        instructions=MENU_INSTRUCTIONS,
    )
    order_agent = Agent(
        name="Order Agent",
        handoff_description="Specialist for taking, updating, and confirming food orders.",
        model=model,
        model_settings=settings,
        instructions=ORDER_INSTRUCTIONS,
    )
    reservation_agent = Agent(
        name="Reservation Agent",
        handoff_description="Specialist for table reservations and booking details.",
        model=model,
        model_settings=settings,
        instructions=RESERVATION_INSTRUCTIONS,
    )

    return Agent(
        name="Triage Agent",
        model=model,
        model_settings=settings,
        instructions=TRIAGE_INSTRUCTIONS,
        handoffs=[
            handoff(
                menu_agent,
                tool_name_override="transfer_to_menu_agent",
                tool_description_override="Route menu, ingredient, allergy, and dietary questions to the Menu Agent.",
            ),
            handoff(
                order_agent,
                tool_name_override="transfer_to_order_agent",
                tool_description_override="Route food ordering, order changes, and order confirmations to the Order Agent.",
            ),
            handoff(
                reservation_agent,
                tool_name_override="transfer_to_reservation_agent",
                tool_description_override="Route reservation and table booking requests to the Reservation Agent.",
            ),
        ],
    )


def initialize_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"restaurant-{uuid.uuid4().hex}"
    if "agent_session" not in st.session_state:
        st.session_state.agent_session = SQLiteSession(
            st.session_state.session_id,
            str(DB_PATH),
        )
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "안녕하세요. 예약, 메뉴, 주문 중 무엇을 도와드릴까요?",
            }
        ]
    if "last_events" not in st.session_state:
        st.session_state.last_events = []


def new_chat() -> None:
    st.session_state.session_id = f"restaurant-{uuid.uuid4().hex}"
    st.session_state.agent_session = SQLiteSession(
        st.session_state.session_id,
        str(DB_PATH),
    )
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "새 대화를 시작했어요. 예약, 메뉴, 주문 중 무엇을 도와드릴까요?",
        }
    ]
    st.session_state.last_events = []


def render_events(events: list[dict[str, str]]) -> None:
    if not events:
        st.caption("handoff 이벤트가 아직 없습니다.")
        return

    for event in events:
        st.markdown(f"- `{event['time']}` {event['message']}")


def extract_handoffs(result) -> list[str]:
    handoffs: list[str] = []
    for item in getattr(result, "new_items", []):
        if isinstance(item, HandoffOutputItem):
            handoffs.append(f"{item.source_agent.name} → {item.target_agent.name}")
    return handoffs


def run_restaurant_bot(prompt: str, model_name: str, api_key: str) -> tuple[str, list[dict[str, str]], dict[str, object]]:
    started_at = time.perf_counter()
    logger = HandoffLogger(started_at)
    agent = build_agents(model_name, api_key)

    result = Runner.run_sync(
        agent,
        prompt,
        session=st.session_state.agent_session,
        hooks=logger,
        max_turns=6,
    )
    elapsed = time.perf_counter() - started_at
    handoffs = extract_handoffs(result)
    final_agent = getattr(result, "last_agent", None)

    evidence = {
        "model": model_name,
        "elapsed": elapsed,
        "handoffs": handoffs,
        "final_agent": getattr(final_agent, "name", "Unknown"),
    }
    return str(result.final_output), logger.events, evidence


def render_sidebar() -> str:
    with st.sidebar:
        st.header("설정")
        model_label = st.radio(
            "Model",
            list(MODEL_OPTIONS.values()),
            horizontal=True,
            label_visibility="collapsed",
        )
        selected_model = next(
            model for model, label in MODEL_OPTIONS.items() if label == model_label
        )
        st.caption("기본은 비용 효율이 좋은 DeepSeek V4 Flash입니다.")

        if st.button("새 대화", use_container_width=True):
            new_chat()
            st.rerun()

        st.divider()
        st.subheader("전문 에이전트")
        st.markdown(
            """
- Triage Agent: 요청 분류
- Menu Agent: 메뉴·재료·알레르기
- Order Agent: 주문 접수·확인
- Reservation Agent: 예약 처리
""".strip()
        )

    return selected_model


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🍽️", layout="centered")
    set_tracing_disabled(True)
    initialize_state()

    model_name = render_sidebar()
    api_key = load_deepseek_api_key()

    st.title(APP_TITLE)
    st.caption("OpenAI Agents SDK handoff 기능으로 요청을 전문 레스토랑 에이전트에게 연결합니다.")

    if not api_key:
        st.warning("DEEPSEEK_API_KEY를 환경변수 또는 Streamlit Secrets에 등록해 주세요. 키 값은 표시하지 않습니다.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.expander("최근 handoff 실행 로그", expanded=bool(st.session_state.last_events)):
        render_events(st.session_state.last_events)

    prompt = st.chat_input("예: 오늘 저녁 7시에 4명 예약하고 싶어요")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        log_placeholder = st.empty()
        response_placeholder = st.empty()
        evidence_placeholder = st.empty()

        with st.spinner("Triage Agent가 요청을 분석하고 있습니다..."):
            try:
                response, events, evidence = run_restaurant_bot(prompt, model_name, api_key)
            except Exception as exc:
                response = (
                    "응답 생성 중 오류가 발생했어요. API 키, 모델 이름, 네트워크 상태를 확인해 주세요.\n\n"
                    f"오류 유형: `{type(exc).__name__}`"
                )
                events = []
                evidence = {
                    "model": model_name,
                    "elapsed": 0,
                    "handoffs": [],
                    "final_agent": "Error",
                }

        st.session_state.last_events = events
        log_placeholder.markdown("**handoff 실행 로그**")
        with log_placeholder.container():
            st.markdown("**handoff 실행 로그**")
            render_events(events)

        response_placeholder.markdown(response)
        handoff_summary = " → ".join(evidence.get("handoffs") or ["handoff 없음"])
        evidence_placeholder.caption(
            f"모델: {MODEL_OPTIONS.get(model_name, model_name)} · "
            f"최종 에이전트: {evidence.get('final_agent')} · "
            f"handoff: {handoff_summary} · "
            f"총 {float(evidence.get('elapsed') or 0):.2f}s"
        )

    st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
