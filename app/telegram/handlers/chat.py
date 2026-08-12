"""일반 메시지 → RAG 답변 핸들러."""

from enum import Enum, auto
from telegram import Update
from telegram.ext import ContextTypes

from app.llm.agent import ask_llm
from app.rag.pipeline import RAGPipeline


# ──────────────────────────────────────────────
# 의도 라벨
# ──────────────────────────────────────────────

class Intent(Enum):
    """메시지 의도 분류."""
    CHITCHAT = auto()   # 인사 / 잡담 → Sources 없음
    GENERAL = auto()    # 비문서 질문 (파일시스템/코드) → Sources 없음
    RAG = auto()        # 문서/지식 질문 → RAG 파이프라인 사용 + Sources


# ──────────────────────────────────────────────
# 키워드 목록
# ──────────────────────────────────────────────

CHITCHAT_KEYWORDS = [
    "안녕", "하이", "헬로", "hello", "hi",
    "안녕하세요", "반갑습니다", "만나서 반가워",
    "감사합니다", "고마워", "수고하세요", "수고해",
    "잘 가", "안녕히 가세요", "안녕히 계세요",
    "미안해", "죄송합니다", "sorry", "thanks",
    "네", "아니요", "좋아", "싫어",
    "응", "응해", "아", "음", "흠",
    "어떻게 지내", "오늘 뭐해", "배경이 뭐야", "누구야",
    "자기소개", "너는 누구", "무슨 로봇", "로봇이니",
    "재미있", "지루해", "힘들어", "수고했어",
    "잘 알아", "모르겠어", "궁금해",
    "how are you", "what's up", "nice to meet", "thank you",
    "tell me about yourself", "who are you", "you are a bot",
]

NON_DOC_KEYWORDS = [
    "project", "folder", "directory", "code", "function",
    "tree", "ls ", "python ", "shell ", "command ",
    "폴더에", "디렉토리에", "코드에", "함수에",
    "파일 목록", "파일 삭제", "파일 생성",
    # 일반 질문 패턴 (문서 외)
    "뭐야", "무슨 소리", "왜", "어떻게 해",
    "알려줘", "기억해", "해석해", "번역해",
    "추천해", "비교해", "설명해줘",
]

DOC_KEYWORDS = [
    # 문서/파일 직접 언급
    "pdf", "문서", "파일에", "문서에", "이 문서",
    "요약", "summarize", "summary", "explain",
    # "~에 대한" 구조 (문서 관련)
    "에 대한", "에관한", "에 관한", "에따라", "에 따라",
    "에대한",
    # "~이란/이라는" 구조
    "이란", "라는",
]


# ──────────────────────────────────────────────
# 라우팅 로직
# ──────────────────────────────────────────────

def _classify_intent(question: str, has_pipeline: bool) -> Intent:
    """질문의 의도를 분류. pipeline 없으면 무조건 GENERAL."""
    if not has_pipeline:
        return Intent.GENERAL

    lower = question.lower().strip()

    if any(k in lower for k in CHITCHAT_KEYWORDS):
        return Intent.CHITCHAT
    if any(k in lower for k in NON_DOC_KEYWORDS):
        return Intent.GENERAL
    if any(k in lower for k in DOC_KEYWORDS):
        return Intent.RAG

    # 기본값: 문서 질문이 아니면 GENERAL (Sources 없음)
    return Intent.GENERAL


def answer_question(
    question: str,
    intent: Intent,
    pipeline: RAGPipeline | None = None,
    llm=ask_llm,
) -> str:
    """의도에 따라 답변 생성.

    - CHITCHAT / GENERAL → pure LLM 응답 (Sources 없음)
    - RAG                 → RAG 파이프라인으로 답변 + Sources 붙임
    """
    if intent in (Intent.CHITCHAT, Intent.GENERAL):
        # Sources 없이 순수 LLM 답변만 반환
        return llm(question)

    # ── RAG 경로 ──────────────────────────────
    if not pipeline:
        print("[answer_question] pipeline is None → fallback to LLM")
        return llm(question)

    try:
        documents = pipeline.retrieve(question)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise

    print(f"[answer_question] retrieve returned {len(documents)} documents")
    if not documents:
        print("[answer_question] no documents → fallback to LLM (no Sources)")
        return llm(question)

    prompt = pipeline.prompt_builder.build(question, documents)
    answer = llm(prompt)

    # RAG로 답변했으므로 _format_answer 로 Sources 추가
    return pipeline._format_answer(answer, documents)


async def handle_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE, pipeline: RAGPipeline
) -> None:
    if not update.message or not update.message.text:
        return

    question = update.message.text

    # 1) 의도 분류
    intent = _classify_intent(question, pipeline is not None)

    # 2) 라우팅
    if intent == Intent.RAG:
        print(f"[handle_message] RAG → {question}")
        answer = answer_question(question, intent=intent, pipeline=pipeline)
    else:
        # CHITCHAT / GENERAL: Sources 없이 LLM 대화만
        print(f"[handle_message] {intent.name} → direct LLM (no Sources): {question}")
        answer = answer_question(question, intent=intent, llm=ask_llm)

    await update.message.reply_text(answer)
