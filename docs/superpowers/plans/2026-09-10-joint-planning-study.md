# Joint Planning Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 共同4方針と独立4方針の最終コード正答率を、固定された既存8 GPU上で、人手採点・費用評価なしに比較できる実験系を作る。

**Architecture:** 既存Qwen 4 workerとDeepSeek 1 workerを研究用HTTP driverから呼び、方針だけを機械的に分割する。生成入力とgoldを分離し、既存機械のCPU containerで公式ベンチマークのテストを実行する。研究条件・途中出力・再開状態・解析を固定する。

**Tech Stack:** Python 3.11.13、httpx、stdlib JSON/asyncio、既存vLLM endpoints、pinned LiveCodeBench/EvalPlus、Docker CPU runner、NumPy。

**Spec:** [研究計画全体](../../../research/joint_planning/README.md)、[protocol.json](../../../research/joint_planning/protocol.json)、[sources.lock.json](../../../research/joint_planning/sources.lock.json)。

## Global Constraints

- 既存8×RTX PRO 6000 Blackwell 96GBと現在のPCIe構成だけを使う。モデル常駐構成を変えない。
- 人手採点・最終評価のLLM judge・費用・費用対効果・GPU時間/token効率・速度ランキングを導入しない。
- 主比較はC_joint対B_independentの最終全テスト成功率。参照条件で主仮説を増やさない。
- gold、テスト結果、canonical solutionを生成処理へ渡さない。
- production example・現在のservingを変更しない。driver用containerはGPUを要求せず既存networkに参加する。
- 本文中のCLIは**実装するインターフェース**。この計画作成時には `cli.py`、driver、採点adapter、Dockerfileは存在しない。以下のチェックが通る前に実行可能と表示しない。
- 研究計画・設定・prompt以外の実装やGPU実験は、この計画作成ターンでは実施していない。

## ファイルと責務

| 作成するファイル | 責務 |
|---|---|
| `research/joint_planning/prepare.py` | pinned取得、payload検証、split、goldを除いた入力export |
| `research/joint_planning/wire.py` | 既存templateに合わせたHTTP payload、seed、回答/thinking分離 |
| `research/joint_planning/plans.py` | JSON方針解析、code fence抽出 |
| `research/joint_planning/run.py` | 五方式の実行、順序、fallback、retry、再開 |
| `research/joint_planning/plansearch.py` | 原方式の主要部と明示した探索幅制限 |
| `research/joint_planning/grade.py` | gold専用の公式実行テストadapter |
| `research/joint_planning/analyze.py` | 対応あり主比較・副指標・完全性検査 |
| `research/joint_planning/attest.py` | source設定とlive workerの同一性、freeze |
| `research/joint_planning/cli.py` | 下記CLIへの入口 |
| `research/joint_planning/Dockerfile.driver`, `Dockerfile.grade` | 生成と採点を分離したCPU環境 |
| `research/joint_planning/requirements.in`, `requirements.lock` | wheel/hashを固定する環境定義 |
| `tests/research/test_joint_planning_*.py` | 各境界の契約検査 |

### Task 1: データ固定と漏洩を防ぐ入力export

**Files:** `prepare.py`, `cli.py`, `tests/research/test_joint_planning_data.py`。

**Interfaces:** `prepare(sources_lock: dict, protocol: dict, cache_dir: Path, out_dir: Path) -> Path` は `out_dir / 'manifest.json'` を返す。取得・検証・exportの具体的契約は以下のチェック項目と研究計画§6に従う。

```python
def split_ids(rows: list[dict]) -> dict[str, list[str]]:
    ranked = sorted(
        (row['platform'] + ':' + row['question_id'] for row in rows),
        key=lambda uid: (hashlib.sha256(
            ('kairyu-joint-planning-v1:split:' + uid).encode()
        ).hexdigest(), uid),
    )
    if len(ranked) != 1055 or len(set(ranked)) != 1055:
        raise ValueError('population identity mismatch')
    return {'dev': ranked[:50], 'test': ranked[50:450], 'unused': ranked[450:]}
```

- [ ] `.part`へstream downloadし、全ファイルのSHA256とbytesが一致した場合だけrenameする。urlはsource lockのrevision/配布版から構成し、`latest`を使わない。
- [ ] JSONLは1行ずつ読み、LCB split manifestにID・本文hash・platform・difficultyを保存する。生成exportは `uid,dataset,split,problem` の4キーだけとする。problemはLCBの `question_content`＋starter code、HumanEval+のpromptから構成する。
- [ ] 生成用に採点器の `CodeGenerationProblem` を作らない。その初期化はprivate testsを復号するため、採点専用側で使う。
- [ ] `manifest.json`は入力sourceとprotocol hashを持つ。異なる内容の既存manifestがあれば上書きを拒否する。
- [ ] 次の境界検査を作り、失敗→実装→成功を確認する。canonical/test列へsentinelを入れてもexportに含まれないこと、ID重複・件数違い・hash不一致を拒否すること、入力行順を変えてもsplitが一致すること。

```python
def test_gold_never_reaches_generation_export(tmp_path):
    row = {'question_content': 'Add two integers.', 'starter_code': '',
           'private_test_cases': 'SECRET_GOLD_SENTINEL',
           'canonical_solution': 'SECRET_REFERENCE_SENTINEL'}
    exported = make_generation_problem(row, dataset='lcb')
    assert 'SECRET_' not in exported
    assert exported == 'Add two integers.'
```

`make_generation_problem(row: dict, dataset: str) -> str`を `prepare.py` に定義する。入力allowlist以外を参照しない。

```python
def make_generation_problem(row: dict, dataset: str) -> str:
    if dataset == 'humaneval_plus':
        return row['prompt']
    if dataset != 'lcb':
        raise ValueError('unknown dataset')
    result = row['question_content']
    starter = row.get('starter_code', '')
    if starter:
        result += '\n\nSTARTER CODE\n```python\n' + starter + '\n```'
    return result
```

Run: `UV_CACHE_DIR=/private/tmp/kairyu-joint-uv uv run pytest tests/research/test_joint_planning_data.py -q`。

### Task 2: モデルwireと厳密な出力境界

**Files:** `wire.py`, `plans.py`, `tests/research/test_joint_planning_wire.py`。

**Interfaces:** `build_payload(role, prompt, seed, caps, protocol) -> dict`、`parse_plans(text, expected) -> list[str | None]`、`extract_code(text) -> str`、`RoleClient.generate(call: dict) -> dict`。generateの返却キーは `content,reasoning_content,raw_response,finish_reason,technical_status` とする。

```python
def parse_plans(text: str, expected: int) -> list[str | None]:
    try:
        result = json.loads(text.strip())
    except (ValueError, TypeError):
        return [None] * expected
    if not (isinstance(result, list) and len(result) == expected
            and all(isinstance(x, str) and x.strip() for x in result)):
        return [None] * expected
    return [x.strip() for x in result]

def extract_code(text: str) -> str:
    match = re.search(r'```[^\n]*\n(.*?)```', text, flags=re.S)
    return match.group(1).strip() if match else ''
```

- [ ] `protocol.json`とpromptファイルからrequestを構成する。DeepSeekの単一user messageに完全scaffoldを入れ、Qwenに通常のrole本文を入れる。具体的wireは研究計画§4に従う。
- [ ] `httpx.MockTransport`で、DeepSeekに二つ目のmessageを作らないこと、thinkingをplanへ渡さないこと、全armでsampling/role capを明示することを検証する。
- [ ] solver requestに同じ問題と指定slotの方針だけが含まれることを、他slotにsentinelを置いて確認する。A/fallbackのassigned_planは空文字。
- [ ] code fenceが閉じていない・ない場合は空コード。複数コードは最初だけ。意味的な修正や追加生成を行わない。
- [ ] `.content`と`reasoning_content`の分離がservingのwireと合わない場合はtechnical failure。別のarmだけ独自にreasoningを剥がす処理を導入しない。

Run: `UV_CACHE_DIR=/private/tmp/kairyu-joint-uv uv run pytest tests/research/test_joint_planning_wire.py -q`。

### Task 3: 四候補を確定するPlanSearch適用版

**Files:** `plansearch.py`, `vendor/plansearch/`, `tests/research/test_joint_planning_plansearch.py`。

**Interfaces:** `async plansearch_candidates(task, replicate_seed, client, logger) -> list[dict]`。4slotを返し、各slotは `index,plan,pseudocode,code,status` を持つ。共通統合は `run.py` が行う。

- [ ] upstream commitとCodeRM submodule commitを確認し、研究計画§5のプロンプト関数とfew-shot参照を取り込む。出典・LICENSE・ファイルhashを保存する。upstreamのGPU依存をinstallしない。
- [ ] 第1層→hash選択した1集合の第2層→探索済み2集合をhash選択→各々原方針/批判/修正→各4slotの擬似コード/コードの順を実装する。**コード生成前に**選択を確定する。

```python
def select_observation_sets(sets, study_id, uid, seed, stage, limit):
    unique = sorted(set(tuple(x) for x in sets))
    def key(obs):
        payload = json.dumps([study_id, uid, seed, stage, obs],
                             ensure_ascii=False, separators=(',', ':'))
        return hashlib.sha256(payload.encode()).hexdigest(), obs
    return sorted(unique, key=key)[:limit]
```

- [ ] 公式の `eval()`を `ast.literal_eval()`＋`list[str]`検査に置換する。list形式化の試行は最大3回、全試行をlogへ入れる。観察不足を重複で埋めず、対応するslotを方針なしsolverへ渡す。
- [ ] Mockで正解テストの有無に依存せず同じ観察集合を選ぶこと、正常経路で4コードが生成されること、形式化初回成功なら統合を含め23呼出しとなること、最大形式化試行でも27論理呼出し以内となることを確認する。infra retryは論理呼出しと別カウンタにする。
- [ ] `num_completions=4`で全コード生成後に切り出す実装になっていないことを、未選択leafの生成を呼ぶと失敗するMockで確認する。

Run: `UV_CACHE_DIR=/private/tmp/kairyu-joint-uv uv run pytest tests/research/test_joint_planning_plansearch.py -q`。

### Task 4: 五方式・完全な記録・再開

**Files:** `run.py`, `tests/research/test_joint_planning_run.py`。

**Interfaces:** `async run_workflow(task, arm, replicate_seed, client, store) -> dict`。返却は `workflow_id,uid,dataset,arm,replicate_seed,candidates,final,technical_status`。storeはJSONLへのappendと `(call_id,attempt)`による再開照合を提供する。

```python
# B: four requests are submitted before awaiting their completion.
plans = await asyncio.gather(*[
    client.generate(make_plan_call(task, replicate_seed, index))
    for index in range(4)
])
```

`make_plan_call(task: dict, replicate_seed: int, index: int) -> dict`を `run.py` に定義し、Task 2のpayload構築・seed規則を使用する。Cは1callの4slotを `parse_plans`で分割する。

- [ ] A/B/C/D/Eを研究計画通りに実装し、候補4slotを保つ。Eは候補配列[]で同一final promptを使う。
- [ ] Bの4plannerが全て開始するまで応答を返さないMockで、並行投入を確認する。4solverも同様。外側workflowは1に固定する。
- [ ] task×seed単位のarm shuffle、worker回転、候補順shuffleとseed派生を保存する。同じblockでarmごとに候補順を変えない。
- [ ] request intentを送信前に記録する。応答・finish reason・全候補・最終回答を切り詰めず保存する。request hash違いでresumeしない。既存の成功/モデル失敗を再生成しない。
- [ ] 研究計画のinfra retryだけを実装する。途中切断で実行状態が不明なら次blockへ進めず、quiescence確認へ移る。再開時はlive attestationも照合する。
- [ ] 空計画→元問題のみで解くfallback、空最終回答→0点、途中打ち切りだが完全コードあり→通常採点、という区別をMockで検証する。

Run: `UV_CACHE_DIR=/private/tmp/kairyu-joint-uv uv run pytest tests/research/test_joint_planning_run.py -q`。

### Task 5: 公式採点器と固定解析

**Files:** `grade.py`, `analyze.py`, `Dockerfile.grade`, `requirements.in`, `requirements.lock`, `tests/research/test_joint_planning_grade.py`, `test_joint_planning_analysis.py`。

**Interfaces:** `grade_saved_outputs(manifest, runs, gold_dir) -> Path`は `grades.jsonl`を返す。各行は `uid,dataset,arm,replicate_seed,stage,slot,pass,reason`。`analyze(manifest, freeze, grades) -> dict`は主比較・副指標・完全性を返す。

- [ ] Python3.11.13環境にpinned scorer sourceを配置する。LCBの評価用importだけを使い、同リポジトリのvLLM/Torchをworkerへinstallしない。依存lockとbuilt image digestをfreezeする。
- [ ] LCBは公式の公開＋private testと6秒/ケース、HumanEval+は公式base＋plusを使用する。参照コード・採点器自身のエラーと、提出コードの失敗を区別する。
- [ ] HumanEval+の完全moduleは `task_id,solution` 形式で公式採点器へ渡す。`completion`によるpromptの二重前置を避け、`min_time_limit=0.2, gt_time_limit_factor=4.0`と共通の保存済み参照時間を使う。
- [ ] ネットワーク/GPUなしの採点containerで既知の正解、誤答、構文不正、無限ループを検証する。正解の参照コードが資源不足の場合、正式試験の失敗として進めない。

```python
def official_results_pass(results):
    return bool(results) and all(value > 0 for value in results)

def test_negative_codes_are_not_success():
    assert not official_results_pass([-1])
    assert not official_results_pass([-2])
    assert not official_results_pass([])
    assert official_results_pass([True, True])
```

- [ ] 全予定workflowと全候補slotの完全性を検査し、重複・未知ID・違うfreeze・不足行を黙って無視しない。
- [ ] `d_i`と問題cluster bootstrapを研究計画§8通りに実装する。2 seedと全armを同じresample indexで引く。常に同じ2 seedを持つ問題が崩れたら失敗する検査を入れる。
- [ ] 分母0のdiscard/rescueはNA。未解決infraを含む全割当結果と、該当問題cluster全体を除いた補助結果を区別する。
- [ ] 出力は正答率、差の区間、候補数分布、遷移表、technical failures。費用や速度を成績列へ追加しない。

Run: `UV_CACHE_DIR=/private/tmp/kairyu-joint-uv uv run pytest tests/research/test_joint_planning_grade.py tests/research/test_joint_planning_analysis.py -q`。

### Task 6: 稼働確認・freeze・実行入口

**Files:** `attest.py`, `cli.py`, `Dockerfile.driver`, `tests/research/test_joint_planning_attest.py`。

**Interfaces:** `attest(worker_inventory) -> dict`、`freeze(protocol, manifest, source_hashes, attestation, cap_level) -> dict`。snapshotが取得不能・不一致の場合は値を推測せず失敗させる。

- [ ] Compose containerの実network名と5workerを取得する。driverはそのnetworkへ入れ、5workerの設定・実argv・GPU UUID・template読込同一性を記録する。秘密を含むenv全体を出力しない。
- [ ] sourceのimage digestだけでなく、実container image ID、StartedAt、model mount/revision、templateのhashと読込時点を確認する。
- [ ] exampleの参照ファイルは `sources.lock.json` の `local_source_commit` から取得して `local_source_sha256` と照合する。mainの同名ファイルを同じ版とみなさず、ソースの一致とlive attestationを別に確認する。
- [ ] freezeにデータ・prompt・移植したfew-shot・採点器・sampling・上限・seed規則・fallback・retry・解析・image/依存lockを含める。cap level未選択、LCB payload未検証、live attestation未確認の状態で本試験を開始できないようにする。
- [ ] CLIのhelpに下記の引数と、準備/dev/freeze/testの順序を実装する。helpだけの架空の動作で完了としない。

```text
python -m research.joint_planning.cli prepare --protocol research/joint_planning/protocol.json --sources research/joint_planning/sources.lock.json --cache research/joint_planning/cache --out research/joint_planning/runs/prepared
python -m research.joint_planning.cli attest --protocol research/joint_planning/protocol.json --out research/joint_planning/runs/attestation.json
python -m research.joint_planning.cli calibrate --prepared research/joint_planning/runs/prepared --protocol research/joint_planning/protocol.json --out research/joint_planning/runs/dev
python -m research.joint_planning.cli freeze --prepared research/joint_planning/runs/prepared --dev research/joint_planning/runs/dev --attestation research/joint_planning/runs/attestation.json --out research/joint_planning/runs/freeze.json
python -m research.joint_planning.cli run --freeze research/joint_planning/runs/freeze.json --dataset lcb --split test --out research/joint_planning/runs/lcb-test
python -m research.joint_planning.cli run --freeze research/joint_planning/runs/freeze.json --dataset humaneval_plus --split confirmation --out research/joint_planning/runs/he-confirmation
python -m research.joint_planning.cli grade --freeze research/joint_planning/runs/freeze.json --runs research/joint_planning/runs --out research/joint_planning/runs/grades.jsonl
python -m research.joint_planning.cli analyze --freeze research/joint_planning/runs/freeze.json --grades research/joint_planning/runs/grades.jsonl --out research/joint_planning/runs/report
```

これらは各専用container内の同一repo mountを基準にしたインターフェース。prepare/grade/analyzeのうちgoldを読む処理をgenerator container内で実行しない。実際のDocker起動コマンドはTask 6で取得したnetwork・built image ID・mount先から出力し、別環境の名前を推測して固定しない。

## 実行順序と検証

- [ ] Tasks 1–6のCPU/Mock検査とlintを実行する。必要な検証対象は `tests/research/test_joint_planning_*.py` と新規研究moduleに限定する。
- [ ] 採点containerの既知正解/失敗試験を実行し、公式scorerとadapterの判定を一致させる。
- [ ] dev最初10問×全5方式×2 seedでcapを機械的に選ぶ。残り40問の確認まで完了する。成績差を理由にpromptを最適化しない。
- [ ] live attestationとfreezeを保存する。正式試験の4,656 workflowを開始するのはこの後。
- [ ] 未解決infra>1%なら実行基盤を点検し、確認的結論を保留する。モデル不正解を理由に問題を補充したりseedを増やしたりしない。
- [ ] 予定行数・ID・hash・全結果の集計を確認し、研究計画全体と結果から許される主張を一緒に報告する。

## 計画作成時の自己レビュー

- 問いと主比較: 最終正答率C−Bに一致。候補の副指標だけで主仮説の成功としない。
- ユーザー制約: ハードウェア固定、人手評価0、費用評価なしをprotocolと全体計画に反映。
- 原実装との差: HTTP driver、担当方針のみ、簡略DAG、PlanSearchの枝刈りを明記。
- 未実施を識別: LCB payload取得/driver/adapter/live attestation/GPU試行を完成済みと書いていない。
- 全体計画の変更時は `README.md`、`protocol.json`、この作業計画を同時に更新する。
