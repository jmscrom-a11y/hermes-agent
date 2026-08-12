"""벡터 인덱스 재생성 스크립트

Ollama 가 실행된 상태에서 사용해야 합니다:
  ollama serve &
  python3 rebuild_index.py

동작:
1. data/docs/ 의 모든 문서 로드
2. 임베딩 생성 (Ollama)
3. FAISS 벡터 인덱스 재생성
4. 기존 데이터/docs/faiss_index/index.* 파일 교체
"""

import os
from pathlib import Path

# 프로젝트 루트를 sys.path 에 추가
sys_path = str(Path(__file__).parent)
if sys_path not in os.sys.path:
    os.sys.path.insert(0, sys_path)

from app.rag.loader import DEFAULT_ROOT_DIR, collect_files
from app.rag.pipeline import build_index
from hermes_v4.config.settings import get_settings


def rebuild_index(docs_dir=None, index_dir="data/faiss_index"):
    """문서를 로드하고 벡터 인덱스를 재생성합니다.

    app.rag.pipeline.build_index() 를 그대로 사용 — 예전엔 여기서
    load_documents() 결과를 split_documents() 청킹 없이 바로
    create_vector_db() 에 넘겨서, build_index.py로 만든 인덱스와
    청크 단위가 달라지는(문서 전체가 하나의 벡터가 되는) 버그가 있었다.
    또한 임베딩 모델도 지정 없이 기본값(nomic-embed-text)을 쓰고 있어서,
    hermes_v4의 RAG_EMBEDDING_MODEL(bge-m3)로 만든 기존 인덱스와 차원이
    달라질 위험이 있었다.
    """

    # docs_dir 이 None 이거나 상대 경로이면 DEFAULT_ROOT_DIR 사용
    if docs_dir is None:
        docs_path = DEFAULT_ROOT_DIR
    else:
        docs_path = Path(docs_dir)
        if not docs_path.is_absolute():
            docs_path = (Path(__file__).parent / docs_path).resolve()

    # build_index.py와 동일하게 hermes_v4 설정에서 임베딩 모델을 읽는다 —
    # RAGTool이 로드할 때 쓰는 모델과 달라지면 FAISS 차원 불일치로
    # 조용히 깨진다 (build_index.py 참고).
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", get_settings().RAG_EMBEDDING_MODEL)

    print("[1/3] 문서 탐색...")
    print(f"   탐색 디렉토리: {docs_path}")
    paths = [str(p) for p in collect_files([str(docs_path)])]
    print(f"  → {len(paths)} 개의 파일 발견")

    if not paths:
        print("ERROR: 로드된 문서가 없습니다.")
        return False

    print(f"[2/3] 임베딩({embed_model}) + 청킹 + 인덱스 생성 중... (Ollama 연결 필요)")
    build_index(paths, index_dir=index_dir, embedding_model=embed_model)

    print(f"[3/3] 인덱스 저장 완료 ({index_dir})")
    print("\n✅ 벡터 인덱스 재생성 완료!")
    print(f"   위치: {index_dir}/index.faiss, index.pkl")
    return True


if __name__ == "__main__":
    rebuild_index()
