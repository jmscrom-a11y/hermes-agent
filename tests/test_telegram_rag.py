import unittest

from app.telegram.bot import answer_question
from app.telegram.handlers.chat import Intent


class FakePromptBuilder:
    def build(self, question, documents):
        return f"question={question}; context={documents[0].page_content}"


class FakePipeline:
    def __init__(self, documents):
        self.documents = documents
        self.prompt_builder = FakePromptBuilder()
        self.last_question = None

    def retrieve(self, question):
        self.last_question = question
        return self.documents

    def _format_answer(self, answer, documents):
        return answer


class FakeDocument:
    def __init__(self, page_content):
        self.page_content = page_content


class TelegramRAGTest(unittest.TestCase):
    def test_answer_question_uses_rag_prompt_when_documents_exist(self):
        prompts = []
        pipeline = FakePipeline([FakeDocument("retrieved context")])

        def fake_llm(prompt):
            prompts.append(prompt)
            return "rag answer"

        result = answer_question("hello", intent=Intent.RAG, pipeline=pipeline, llm=fake_llm)

        self.assertEqual("rag answer", result)
        self.assertEqual("hello", pipeline.last_question)
        self.assertEqual("question=hello; context=retrieved context", prompts[0])

    def test_answer_question_falls_back_to_llm_when_no_documents(self):
        prompts = []
        pipeline = FakePipeline([])

        def fake_llm(prompt):
            prompts.append(prompt)
            return "llm answer"

        result = answer_question("hello", intent=Intent.RAG, pipeline=pipeline, llm=fake_llm)

        self.assertEqual("llm answer", result)
        self.assertEqual(["hello"], prompts)

    def test_answer_question_falls_back_to_llm_without_pipeline(self):
        prompts = []

        def fake_llm(prompt):
            prompts.append(prompt)
            return "llm answer"

        result = answer_question("hello", intent=Intent.RAG, pipeline=False, llm=fake_llm)

        self.assertEqual("llm answer", result)
        self.assertEqual(["hello"], prompts)


if __name__ == "__main__":
    unittest.main()
