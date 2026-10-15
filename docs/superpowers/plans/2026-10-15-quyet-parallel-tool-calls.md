# 計画：二つの Quyet オーケストレーションを「並列ツールコール＋Quyet 判定」に共通で作り直す

状態: 承認 2026-10-15（VCO-D24）。ブランチ: `claude/quyet-parallel-tool-calls`。

## 目的と背景

- **問題:** オーケストレーション（verified tool route）の推論設定を max にすると、L3 の 1 コールが長くなる。その結果、エージェント側の 900 秒の切断と、3 時間の時間切れが増えた。
- **今の構造:** 1 ターンで書き手（DeepSeek/GLM）を 2 回呼ぶ（candidates 10〜16 個 → answer）。さらに Quyet を約 5 回読む（ルート判定・候補判定 約 3 回・形式チェック）。
- **実測（止めた max run）:**
  - candidates は L1 出力の 65 %、候補判定は Quyet 入力の 90 % を占めていた。
  - L3 の 1 コールは p50 約 320 秒、最大 892 秒だった。
- **新しい構造（オーナーのコンセプト）:**
  1. 書き手が 1 回で返答を書く。そのとき、いま必要で、安全で、互いの結果を待たないツールコールを、できるだけ一緒に入れる。
  2. Quyet（判定モデル）が返答を判定する。見るのは「まだ並列で足せるコールがあるか」「安全か」「形式要件を満たすか」の 3 点。
  3. NG なら、Quyet の判定結果を見せて書き手に直させる。
- **ねらい:** 1 ターンの時間を短くし、900 秒の切断と時間切れを減らす。あわせて API コール数とトークンを減らす。
- **やらないこと:**
  - フレームワーク（`kairyu/`）は変えない。example の設定・プロンプト・検証だけを変える。
  - ルート判定（TOOL / THINK）と THINK ルートは今のまま。

## 新しい TOOL ルート（両 example で共通）

| 段階 | 担当 | 内容 |
|---|---|---|
| answer | 書き手（DeepSeek / GLM）、呼び出し元の推論設定 | 呼び出し元の会話をそのまま渡し、最後にプロンプトを 1 つ足す。返答は 1 回で書く。 |
| decision | Quyet（GPU 7、strict で読む） | 返答・ツール定義・会話（今と同じ上限）を 1 回で読み、9 問を判定する。合格ラインは各問 p ≥ 0.5。 |
| fix | 書き手 | NG のとき 1 回だけ直す。下書きと、Quyet の判定（NG の項目と確率）を見せる。直した返答は decision で再判定し、結果にかかわらず、最後に書いた返答を返す。 |

- **形式チェックが読めないとき:** 状態が Quyet の上限 6,000 トークンを超えると strict で拒否される。そのときは今と同じく、検証なしで返す。
- **予算:** `max_steps` は 5（answer、判定、修正、再判定、空出力の再送 各 1）。`max_refine_depth` は 1 にする。`internal_max_tokens` は、使う役割がなくなるので削除する。
- **1 ターンあたりの L1 API コール:** 書き手 1 回（修正時は 2 回）と Quyet 2 回（修正時は 3 回）。今は書き手 2 回以上と Quyet 約 5 回。
- **修正を 1 回にする理由:** 時間を短くするのが目的なので、修正 1 回と再判定 1 回までに抑える。
- **GLM の修正:** VCO-D23 の第3修正で修正をなくしていたが、新しいコンセプトに合わせて両方とも修正ありで揃える。
- **GLM の F5 の文言:** 「まず 1 コール」と書いた GLM 専用の文言（VCO-D23 の第2修正）はやめる。コンセプト（並列に入れる）と逆になるので、DeepSeek と共通の文言にする。
- **例外:** F2（書き出してはいけないコールの書式）だけは、書き手ごとの書式を残す。DeepSeek は DSML、GLM は `<arg_key>/<arg_value>`。

### answer のプロンプト（案）

```
Before replying, read this note. It is not part of the conversation: never
mention it in the reply. The reply is the next assistant message: the tool
call or calls for the move needed now, from where the work stands, with at
most a short text; not the final result in one jump, and not a repeat of a
step whose result the conversation already shows. Make this move as large as
it safely can be: include every call that is needed now and can be made now,
that is, every call whose arguments and safety depend only on what the
conversation already shows, not on the result of another call in this reply,
and that changes nothing another call in this reply reads or changes (for
example, read every file the next step needs, or run every independent check,
together). A call that needs another call's result waits for a later move.
Make several calls only when the conversation's instructions do not limit a
reply to one call or one step. Every call must be safe now: it destroys,
overwrites or reverts no work or data the request needs, and changes nothing
the request says must stay as it is. The finishing call (the call or command
the conversation defines for ending the work) goes alone in its reply, and
only after earlier tool results show every step it depends on succeeded. Make
every call through the tool-calling interface; never write a call, its
arguments or a description of one in the reply text, and never announce a
call or an action without making the call in this reply. Write only the
reply, in exactly the format the conversation demands.
```

### decision の 9 問（state：reply、tools、conversation）

各問の ask は、今と同じく「In the state, reply is the draft reply's text followed by its tool calls, each written as `<tool_call>{{...}}</tool_call>`.」で始める。

| ID | 分類 | 命題（proposition） | 合格（criteria_true）の要点 | NG（criteria_false）の要点 |
|---|---|---|---|---|
| R1 | 並列の余地 | The reply makes every call that is needed now and can be made alongside | いま必要で、安全で、この返答の他のコールにも、まだ会話にない結果にも依存しないコールで、抜けているものがない。または会話が 1 返答 1 コール（1 ステップ）に限っている。または返答が終了コールである。 | いま必要で、安全で、独立したコールが抜けている（次の段階で読むべき別のファイル、独立した確認など）。そのせいで返答の回数が必要以上に増える。 |
| S1 | 安全 | Every call is safe to make now | 必要な作業やデータを壊さない・上書きしない・戻さない。要求が保つと言うものを変えない。正しい入力に新しいエラーや警告を足さない。 | 上の条件のどれかを破る。 |
| S2 | 安全 | Batched calls need no result of each other（今の DeepSeek の F5） | 1 コールである。または、どのコールの引数も安全性も会話に既にあるものだけで決まり、会話も複数コールを許している。 | あるコールが同じ返答の別のコールの結果を必要とする。または会話が 1 コールに限っているのに複数コールしている。 |
| S3 | 安全 | A finishing call is alone and follows shown success（今の F6） | 終了コールがない。または、終了コールは単独で、依存する各ステップの成功がツール結果で示された後にある。 | 終了コールが他のコールと一緒になっている。または成功が示される前にある。 |
| F1 | 形式 | Every action the text announces is made as a call（今の F1） | 今と同じ | 今と同じ |
| F2 | 形式 | The text writes out no call（今の F2。書き手ごとの書式） | 今と同じ | 今と同じ |
| F3 | 形式 | Every call names a declared tool with valid arguments（今の F3） | 今と同じ | 今と同じ |
| F4 | 形式 | The reply makes at least one call（今の F4） | 今と同じ | 今と同じ |
| F5 | 形式 | The text is short and mentions no internal material（今の F7。「judgments、drafts、checks」に言及しない） | 今と同じ | 今と同じ |

### fix のプロンプト（案）

```
Before replying, read the material below. It is not part of the
conversation: never mention it in the reply. A draft of the next assistant
message was written, and an independent judge found that it does not meet
some requirements; each unmet requirement is listed with the judge's
probability that it holds. The judge can be wrong: check each finding against
the conversation yourself. The draft follows as text: its tool calls appear
after its text as <tool_call>{{...}}</tool_call> and are not part of its text.
Write the reply again. Change only what the unmet requirements need, and keep
every other call with the same arguments and the same short text:
- R1 (calls missing): add the calls that are needed now, safe now and need no
  result of another call in the reply.
- S1 (a call is not safe now): replace it with a safe call for the same step,
  or leave it out.
- S2 (calls depend on each other): keep the calls that need no other call's
  result and leave the rest for a later move.
- S3 (finishing call batched or premature): leave it out and make the call
  that shows the steps it depends on succeeded.
- F1-F5 (form): fix only the form.
Make every call through the tool-calling interface; never write a call, its
arguments, a <tool_call> envelope or a description of one in the reply text,
and never announce a call or an action without making the call in this reply.
Write only the reply, in exactly the format the conversation demands.
--- DRAFT ---
{previous}
--- END DRAFT ---
--- JUDGE'S FINDINGS ---
{feedback}
--- END ---
```

- `{feedback}` は、フレームワークの仕様で NG の項目だけが確率付きで並ぶ（`kairyu/orchestration/checklist.py:654-680`）。これがオーナーの言う「判定結果も見せて」にあたる。合格した項目まで見せるにはフレームワークの変更が要るので、やらない。

## 変更一覧（すべて example 側。`kairyu/` は変更なし）

| 変更 | 層 | 担当 | ファイル（両 example とも） |
|---|---|---|---|
| TOOL ルートを answer → decision（→ fix 1 回）にする。candidates と judgments を削除する。プロンプトと 9 問、予算、ヘッダーのコメント | L2 | example | `examples/{deepseek-v4.1,glm-5.3-flash}-quyet-8gpu/verified-tool.yaml` |
| 使わなくなった `quyet-judge-systemone`（strict でない方）を削除し、説明文を直す。GPU の割り当ては変えない（GPU 6：ルート判定、GPU 7：decision） | L1/L2 | example | `kairyu.yaml`、`example.json`（serves、budget）、`compose.yaml` のコメント |
| ゲートの判定を新しい段階に合わせる。STAGES = answer, decision。decision は 9 問・`quyet_form`。effort は `{answer}`。修正は 1 回まで。集計に、問ごとの NG 数、修正の前後、1 返答のコール数、1 リクエストの最大時間を足す | 検証 | example | `verification.py` |
| GLM の l1 の文法 JSON 確認は残す（呼び出し元の response_format で今も使うため）。コメントだけ直す | 検証 | example | `glm-5.3-flash-quyet-8gpu/control.py` |
| 単体テストを差し替える（下のテスト方針） | テスト | example | `tests/unit/test_deepseek_v41_quyet_example.py`、`tests/unit/test_glm53_quyet_example.py` |
| README（冒頭、L2 の図、ゲートの表、制約）と MEASUREMENTS（GPU 検証の結果） | 文書 | example | `README.md`、`MEASUREMENTS.md`、`examples/README.md` の 2 行 |
| 設計の決定 VCO-D24 を新しく足す。VCO-D22 の candidates/judgments、VCO-D23 の第2・第3修正を置き換える。Status 行も更新する | 文書 | docs | `docs/design/example-verified-checklist-orchestration.md` |
| Current Status の 2 行と Change Log 1 件（8 → 9 件） | 文書 | docs | `PROGRESS.md` |

## テスト方針（CLAUDE.md のとおり）

- **DeepSeek と GLM の両方:**
  - 「tool route は書き手を 1 回呼び、Quyet が 1 回 9 問で読む」を 1 本で確かめる。確かめる内容は、会話をそのまま渡すこと、呼び出し元の推論設定、strict のモデル、state のキー、返答がそのまま返ること。
  - 「NG の返答は 1 回直されて再判定される」を確かめる。修正が合格する場合と、修正しても NG のまま返る場合の 2 通り。修正には下書きと NG 項目の確率が入ること、同じ会話・ツール・推論設定で呼ぶことを確かめる。
- **GLM:** 「修正なしで返す」テストは、機能がなくなるので削除する。
- **残すもの:** 「長すぎて読めない返答は検証なしで返す」テストと、ルート判定・THINK のテストは残す。candidates 用の偽の書き手は削除する。
- **ローカルで回すもの:** `uv run ruff check .` と、`CUDA_VISIBLE_DEVICES= uv run --frozen pytest` を変更した 2 ファイルと `test_frontier_examplectl.py` だけに対して回す。全体は CI に任せる。
- **テスト数の比較:** main と PR の head で、同じコマンドで集めたテスト数を比べて報告する。

## 進め方

0. **止めた 2並列 run のデータを消す。** 前の 8並列・4並列の中断 run と同じく、結果フォルダ、telemetry、キャッシュ、Docker データ、イメージを消す。
1. **新しいブランチと draft PR を作る。**
   - main から新しいブランチ `claude/quyet-parallel-tool-calls` を作る（指定ブランチは過去の PR の履歴を持つので使わない）。
   - この計画を `docs/superpowers/plans/2026-10-15-quyet-parallel-tool-calls.md` として commit して push し、draft PR を作る。PR 本文の全文を返信に貼る。
2. **実装する。** 上の変更一覧のとおり。
3. **CPU で確かめる。** ruff と上のテストを回してから push し、CI がすべて成功するのを確かめる。
4. **GPU ゲート（DeepSeek）。** DeepSeek+Quyet の example を `./run.sh` で起動し、9 ゲートを順に回す。
5. **GPU ゲート（GLM）。** DeepSeek を止めて GLM+Quyet を起動し、同じく 9 ゲートを回す。
   - 4・5 とも、NG が出たらその場で止める。分かっているリスクをまとめて直し、該当するターンだけで確かめてから、全体を 1 回だけ回し直す。
6. **記録してレビューに出す。** MEASUREMENTS、設計の文書、PROGRESS を更新して push し、CI 成功を確かめてから draft を解除する。オーナーのレビューで指摘がなくなるまで、GPU の作業はしない。マージはオーナーの指示を待つ。
7. **ベンチマークを回す。** マージ後、main で回す（#648 → #649 と同じ流れ。製品として main を測るため）。
   1. DeepSeek+Quyet 8GPU max：kairyu-bench の DeepSWE を全 113 問、8並列、1 回ずつ、推論設定 max で回す。やり直し 3 回、API の上限 3600 秒、専用の DinD と relay の記録は前と同じ。
   2. 終わったら、すぐに wiki に追記する。同時に、自動で GLM を起動する。
   3. GLM+Quyet 8GPU max：同じ条件で回す。終わったら、すぐに wiki に追記する。
   - **wiki での比較先:** DeepSeek+Quyet 8GPU high（前の構造）、DeepSeek 6GPU max、GLM 6GPU max。
   - **比較する数:** 正解数、L3/L1 の API コール数とトークン数、時間、900 秒の切断と 3 時間の時間切れの数。
   - **推論の経路の欄:** 新しい構造（書き手 1 回 → Quyet が判定 → NG なら 1 回修正）を明記する。
   - **進捗の報告:** 毎回すべて載せる。問題ごとの進み具合と、比較先との比較を入れる。
   - **再実行:** 通信障害などの事故があった場合だけ。時間切れは不正解として数える。

## GPU ゲート（両 example とも、`verification.py` の GATES 全 9 件を順に）

| # | ゲート | 合格条件（変わるところは太字） | 時間上限 |
|---|---|---|---|
| 1 | l1 | 今と同じ（書き手の全 DP rank、Quyet 2 台、strict の拒否、公開モデルのコール 1 つ） | — |
| 2 | routing | 今と同じ（TOOL の見逃しが両方とも 10 % 未満、THINK の精度 90 % 以上） | 30 分 |
| 3 | think-route | 今と同じ | 30 分 |
| 4 | effort | 指定なし・low・high・max のそれぞれで、**answer（と fix）が呼び出し元の推論設定で動く**。ルート判定は quyet-route | 90 分 |
| 5 | verified-tool-route | 6 ターン × 通常 / ストリームの 12 件がすべて TOOL で 200。**answer が成功し、decision が quyet_form で 9 問を判定する（strict で読めないときは検証なしの報告）。修正は 1 回まで**。宣言されたツールへの構造化されたコールで、引数は JSON オブジェクト。**1 件も 900 秒を超えない** | 120 分 |
| 6 | fallback | quyet-route を止めると THINK ルートになる。**quyet-judge を止めると、TOOL のまま answer を検証なしで返し、コールは正しい**。両方戻すと、すべて正常に戻る | 60 分 |
| 7 | serving | c1/c4/c8/c16（8/16/16/32 件）がすべて応答し、すべて構造化されたコール。ルートごとの遅延 p50/p95/最大、トークン、tok/s を記録し、前回の値と比べる（DeepSeek：p50 74.8/65.4/108.7/134.2 s、GLM：151.5/111.5/142.5/185.6 s） | 6 時間 |
| 8 | serving-routed | ルーティングセットがすべて応答する。ルートの内訳とルートごとの遅延・トークンを記録する | 6 時間 |
| 9 | browser | Open WebUI が答える | — |

- **ゲートで記録する追加の数字:**
  - decision の各問の NG 数。
  - 修正の結果（合格、NG のまま、コールが減った）。
  - 1 返答のコール数の分布。
  - 1 ターンの L1 コール数とトークン数。前の構造の記録と比べる。
- **見込み時間:** 1 つの example につき 3〜5 時間。

## リスクと対処

- **R1 で足したコールが安全でなくなる:** R1 の NG で直したら、S2 や S3 で NG になることがある。修正は 1 回までなので、その返答がそのまま返る。ゲートで「修正前は合格していた安全の項目が、修正後に NG」になった件数を数えて報告する。目立つときは、R1 だけ合格ラインを下げる案を持ってくる。
- **GLM の S2 の読み違い:** 以前は GLM の F5 で読み違いがあった。共通の文言で、ゲートの S2 の NG 率を確かめる。
- **書き手が 1 回で長く考えすぎる:** max では answer 1 回が長くなる。verified-tool-route の 900 秒の条件と serving の最大時間で確かめる。

## 決定

- 2026-10-15 オーナー承認：上の計画（9 問とプロンプト、修正 1 回、ゲート、ベンチマークの順番）と、verified-tool-route の「1 件も 900 秒を超えない」条件。
- 2026-10-15 オーナー指示「プロンプトは最適なものを考えて作って」：S2 は GLM で再読により確かめた「まず 1 コールか」を問う文言（VCO-D23 第2修正）に両方とも揃えた。R1 は「会話が示している、いま必要なコール」に限定し、推測の誤読で修正が増えないようにした。answer のメモに「いま不要なコールは足さない」を加え、修正のメモに「会話が支持しない指摘は、その部分をそのまま残す」を加えた。
- 2026-10-15 オーナー指示「DSL の構成は変えずに、プロンプトだけで、単体と比べて正解率を下げず、スピードも落とさない。汎用性は保つ」：記録したターンで再生して比べたところ、ツールの結果の後にユーザーのメッセージを 1 つ足すだけで、書き手が作業を考え直すと分かった（出力の p50 は、メモなしで 1,242、メモありで 5,801〜8,210。メモの長さには関係がない）。そこで answer のプロンプトをなくした。要件は decision と修正 1 回が受け持つ。速さと正解率は、新しく始めた DeepSWE の問題で、同じ問題の単体と比べて判断する（20 問の時点で、時間が 1.1 倍を超えるか、正解が 2 問以上少なければ止める）。
