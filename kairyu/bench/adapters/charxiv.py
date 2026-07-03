"""CharXiv Reasoning: chart-image questions, judge-graded (needs vision + judge)."""

from __future__ import annotations

import base64

from kairyu.bench.adapters.base import (
    AdapterInfo,
    DownloadContext,
    GenerativeAdapter,
    RunContext,
    excerpt,
)
from kairyu.bench.judge_prompts import CHARXIV_JUDGE_TEMPLATE
from kairyu.bench.types import (
    BenchItem,
    BenchTarget,
    ChatRequestSpec,
    DatasetUnavailable,
    ItemResult,
    SkipItem,
)

_INSTRUCTION = (
    "Answer the question about the chart. Reply with a short final answer — "
    "a number, a label, or a short phrase — on the last line as "
    '"Final answer: <answer>".'
)


class CharXivAdapter(GenerativeAdapter):
    info = AdapterInfo(
        name="charxiv-reasoning",
        display_name="CharXiv Reasoning",
        metric="accuracy (judge-graded)",
        hf_dataset="princeton-nlp/CharXiv",
        needs_vision=True,
        judge_preferred=True,
    )

    def normalize(self, ctx: DownloadContext) -> list[dict]:
        from kairyu.bench.hub import load_hf_rows, save_asset

        try:
            rows = load_hf_rows(self.info.hf_dataset, split="validation")
        except DatasetUnavailable:
            rows = load_hf_rows(self.info.hf_dataset, split="val")
        assets = ctx.cache.assets_dir(self.info.name)
        normalized = []
        for index, row in enumerate(rows):
            question = row.get("reasoning_q") or row.get("question")
            answer = row.get("reasoning_a") or row.get("answer")
            image = row.get("image")
            if question is None or answer is None or image is None:
                raise DatasetUnavailable(
                    f"{self.info.hf_dataset} format drift: expected "
                    "reasoning_q/reasoning_a/image fields"
                )
            import io

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            name = f"charxiv-{index:05d}.png"
            save_asset(buffer.getvalue(), assets, name)
            normalized.append(
                {
                    "id": f"charxiv-{index:05d}",
                    "question": question,
                    "answer": str(answer),
                    "image": f"assets/{name}",
                }
            )
        return normalized

    def check_preconditions(self, target: BenchTarget, ctx: RunContext) -> str | None:
        if ctx.judge is None:
            return "requires a judge endpoint (--judge-base-url/--judge-model)"
        return super().check_preconditions(target, ctx)

    def _image_url(self, item: BenchItem, ctx: RunContext) -> str:
        ref = item.payload["image"]
        if ref.startswith("data:"):
            return ref
        path = ctx.cache.adapter_dir(self.info.name) / ref
        encoded = base64.b64encode(path.read_bytes()).decode()
        return f"data:image/png;base64,{encoded}"

    def build_request(
        self, item: BenchItem, target: BenchTarget, ctx: RunContext
    ) -> ChatRequestSpec | SkipItem:
        content = [
            {"type": "text", "text": f"{_INSTRUCTION}\n\n{item.payload['question']}"},
            {"type": "image_url", "image_url": {"url": self._image_url(item, ctx)}},
        ]
        return ChatRequestSpec(
            messages=({"role": "user", "content": content},),
            max_tokens=min(target.max_output_tokens, 2048),
        )

    async def score(
        self, item: BenchItem, response_text: str, ctx: RunContext
    ) -> ItemResult:
        verdict = await ctx.judge.grade(
            question=item.payload["question"],
            expected=item.payload["answer"],
            response=response_text,
            template=CHARXIV_JUDGE_TEMPLATE,
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
