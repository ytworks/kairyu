"""Humanity's Last Exam: MCQ exact match + judge-scored free-form (gated HF dataset).

Official scoring uses an LLM judge for the exactMatch split; without a judge
those items are recorded "unjudged" and the pair degrades to partial — never
a fabricated number (user decision 2).
"""

from __future__ import annotations

from kairyu.bench.adapters.base import (
    AdapterInfo,
    DownloadContext,
    GenerativeAdapter,
    RunContext,
    estimate_tokens,
    excerpt,
    extract_choice_letter,
)
from kairyu.bench.types import BenchItem, BenchTarget, ChatRequestSpec, ItemResult, SkipItem

_INSTRUCTION = (
    "Answer the question below. Think step by step if needed, then finish "
    'with a line of the form "Final answer: <your answer>". For multiple-'
    "choice questions the final answer is the letter of the correct choice."
)


class HleAdapter(GenerativeAdapter):
    info = AdapterInfo(
        name="hle",
        display_name="Humanity's Last Exam",
        metric="accuracy",
        hf_dataset="cais/hle",
        gated=True,
        judge_preferred=True,
    )

    def normalize(self, ctx: DownloadContext) -> list[dict]:
        from kairyu.bench.hub import load_hf_rows, save_asset

        rows = load_hf_rows(self.info.hf_dataset, split="test", gated=True)
        normalized = []
        assets = ctx.cache.assets_dir(self.info.name)
        for row in rows:
            image_ref = None
            image = row.get("image")
            if image:  # data-URL string or PIL image depending on datasets version
                if isinstance(image, str):
                    image_ref = image  # already a data URL; embed verbatim
                else:
                    import io

                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    name = f"{row['id']}.png"
                    save_asset(buffer.getvalue(), assets, name)
                    image_ref = f"assets/{name}"
            normalized.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "answer": str(row["answer"]),
                    "answer_type": row.get("answer_type", "exactMatch"),
                    "image": image_ref,
                }
            )
        return normalized

    def _image_url(self, item: BenchItem, ctx: RunContext) -> str | None:
        ref = item.payload.get("image")
        if not ref:
            return None
        if ref.startswith("data:"):
            return ref
        import base64

        path = ctx.cache.adapter_dir(self.info.name) / ref
        encoded = base64.b64encode(path.read_bytes()).decode()
        return f"data:image/png;base64,{encoded}"

    def build_request(
        self, item: BenchItem, target: BenchTarget, ctx: RunContext
    ) -> ChatRequestSpec | SkipItem:
        image_url = self._image_url(item, ctx)
        if image_url is not None and not target.supports_vision:
            return SkipItem(reason="image question on a non-vision target")
        text = f"{_INSTRUCTION}\n\n{item.payload['question']}"
        if image_url is None:
            content: str | list = text
        else:
            content = [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        return ChatRequestSpec(
            messages=({"role": "user", "content": content},),
            max_tokens=target.max_output_tokens,
            est_prompt_tokens=estimate_tokens(text),
        )

    async def score(
        self, item: BenchItem, response_text: str, ctx: RunContext
    ) -> ItemResult:
        payload = item.payload
        if payload["answer_type"] == "multipleChoice":
            extracted = extract_choice_letter(response_text, num_choices=26)
            return ItemResult(
                item_id=item.id,
                status="completed",
                score=1.0 if extracted == payload["answer"].strip().upper() else 0.0,
                response_excerpt=excerpt(response_text),
            )
        if ctx.judge is None:
            return ItemResult(
                item_id=item.id,
                status="unjudged",
                error="free-form answer requires a judge (--judge-base-url/--judge-model)",
                response_excerpt=excerpt(response_text),
            )
        verdict = await ctx.judge.grade(
            question=payload["question"],
            expected=payload["answer"],
            response=response_text,
        )
        if verdict.correct is None:
            return ItemResult(
                item_id=item.id,
                status="unjudged",
                error=f"judge verdict unparseable: {verdict.raw_excerpt!r}",
                response_excerpt=excerpt(response_text),
                judge=verdict.as_dict(),
            )
        return ItemResult(
            item_id=item.id,
            status="completed",
            score=1.0 if verdict.correct else 0.0,
            response_excerpt=excerpt(response_text),
            judge=verdict.as_dict(),
        )
