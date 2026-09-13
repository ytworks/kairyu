# DeepSeek V4.1 6 GPU＋Qwen3.8-27B 2レプリカのアンサンブル実装計画

更新日: 2026-09-14。基準はローカル／リモートで一致を確認した
`main = 99c5eadbc67d32d57823accd89aac34215e2c720`。
2026-09-13にユーザーが本計画を承認し、draft PRを作成して小さな単位でpushしながら実装するよう指示した。
2026-09-14のユーザー指示により、新規 `examples/qwen3.8-deepseek-v4.1-8gpu/` を作る。
既存のV4 ensembleとV4.1 standalone exampleへの変更はスコープ外とし、mainから変更しない。
実装・GPU検証の状態は[実装記録](../../design/v41-critical-ensemble.md)に記録する。

## 1. 目的と今回の基準

1. DeepSeek V4.1 Flashを6 GPUで1サービス、Qwen3.8-27B FP8を1 GPU×2レプリカで稼働させる。
2. mainの既存アンサンブルを基準に、4方針・4 Qwen候補を維持し、独立したDeepSeek候補、批判的検討、再構成、DeepSeekによる監査を組み合わせる。
3. 元入力と各モデルの推論能力を保ち、多角的検討・批判・統合によって単体モデルを超える精度を目指す。

精度向上率の実証や113タスクの再実行は完了条件にしない。要求抽出や監査を実行した事実と、モデルの判断が正しいことは区別する。

### mainと旧計画の差分

| 項目 | 確認したmain／GitHubの状態 | 本計画への反映 |
|---|---|---|
| PR #600 | 境界ルールの追加としてマージ済み。モデル・アンサンブル実装は含まない | open draftとして維持する指示を削除。実装時は新しいbranch／draft PRを使う |
| PR #595／#598 | ともにclosed・未マージ | #595のRequirementの仕様を参照する。旧実装・独自helper・測定結果は新実装へ持ち込まない |
| 既存アンサンブル | `examples/qwen3.8-deepseek-v4-8gpu/`。Qwen TP1×4＋V4 TP4/EP4、5ルート、Qwen audit | DSL・設定の参照元とする。既存example自体は変更しない |
| Requirement | mainには存在しない | #595の要求抽出・ID・監査契約を、DeepSeekで動く新しいDSL roleとして設定する |
| V4.1 runtime | `examples/deepseek-v4.1-flash-8gpu/` のTP8/EP8を実測済み | L1の基準として参照する。6 GPUの成立性・性能は別途検証する |
| 既存ensembleのGPU証拠 | 最終greenはDTO-D8..D14。mainのDTO-D15は再検証待ち | main全体がGPU検証済みという前提を置かない |
| Issue #599 | open。障害の記録は旧branchの`83854e93`に属する | 保存要求をmainの同種経路で再現し、修正後に再実行する。旧6＋2構成がmainに存在すると扱わない |

## 2. 実装と所有権の原則

- mainから実装を組み立てる。ユーザー指定により、クローズ済み#595のRequirementの**仕様**は継承するが、破棄した実装を復元・cherry-pickしない。
- exampleのオーケストレーションはKairyuのDSL、Conductor、ReplicaPool、既存の監査・修正機構で記述する。exampleにPythonファイルを新規作成しない。
- 新exampleは設定と薄いshell入口からCompose・既存共有検証ツールを使う。既存exampleのPython helperを変更・複製せず、新しいPythonファイルや独自推論基盤を追加しない。
- framework修正ゼロを出発点とする。既存拡張点で構成できない共有契約だけを、第7節の採用条件で評価し、最小限修正する。
- モデル・GPU・候補数、Requirementスキーマ、役割prompt、探索順、批判・監査方針、予算方針はexampleが所有する。名前を汎用化しただけのexample固有処理はframeworkへ入れない。
- 原文・候補の圧縮、要約による置換、切り捨て、容量を理由としたルート回避・アンサンブル省略は行わない。
- 境界ruleの本文は[framework-boundary.md](../../../.claude/rules/framework-boundary.md)に一本化済み。`CLAUDE.md`の読み込みと`AGENTS.md`の参照関係を維持し、規則を重複追加しない。

主変更先は新規の`examples/qwen3.8-deepseek-v4.1-8gpu/`とする。既存V4/V4.1 exampleは参照元として保持する。runtime補正が必要なら新exampleのsource hash付きpatchと派生Dockerfileに置き、既存exampleのbuild/helperを変更しない。

## 3. GPU配置とV4.1の接続

| サービス | 配置案 | 候補生成との関係 |
|---|---|---|
| Qwen 0／1 | GPU 6／7、TP1×2 | `answer_1..4`の4候補を同じ2レプリカpoolへ投入する |
| DeepSeek V4.1 | GPU 0–5、1物理サービス | Requirement、独立候補、policies、批判、再構成、監査とDeepSeek直答を担当する |

2026-09-13の事前検証では現稼働のGPU配置を維持する案へ具体化した。TP2／DP3／EP6とCPU Engram offloadは検証候補であり、[数値事前検証の未通過](../../../examples/qwen3.8-deepseek-v4.1-8gpu/MEASUREMENTS.md#v41c-six-gpu-preflight-numerical-gate-fail-2026-09-13)を理由に、フルモデル起動と構成選定は未完了としている。

候補数とレプリカ数を分ける。4方針は`POLICY 1..4`を維持し、4候補を2候補に減らさない。配置は既存ReplicaPoolのqueue-depth／affinity機構に任せ、候補とGPUを固定対応させない。4候補が4 GPUで同時に動く場合と同じ性能は仮定しない。

### 6 GPUの成立性を先に確認する

mainのV4.1基準はcheckpoint `dba1be0a40aa45a94ad051997016db3960a90277`、runtime image
`sha256:027bf47b2bd6f0d0abe54b296e7e9e3d31ee103bb6e46fa0a9807117681c2359`。
実測済みなのはTP8/EP8、DSpark 5、batch 16384、max sequences 64、NCCL、GPU上のEngramである。
SM120の64-token manager/SWA block、BLHNC、FP8 MLA KV、MXFP4 indexer等の既存適合も基準に含める。

この設定のGPU数を6へ変えただけでは成立しない。採用する並列構成は、固定revisionのモデル／runtimeについて次を確認してから決める。

- Attention・TP・Attention-DP・EPの分割条件、targetとDSpark draft双方のexpert配置。
- weights、Engram、KV、activation、CUDA Graphのメモリと必要なhost memory。
- startup、実リクエスト、kernel／collectiveの正しさ、取消、通常再起動。
- 6 GPUでの長文・同時実行時の容量と性能。offloadやDSpark変更が必要なら、その設定で測り直す。

TP6やPPの組合せ、DSpark 5の継続を先に動作保証しない。mainにない分散runtimeを本計画で新造することも前提にしない。要件を満たす既存runtime構成が成立しなければ、その障害を残し、6＋2構成を実装完了としない。

### native会話・画像・thinking

- 旧ensembleのDeepSeekはV4用scaffoldと`/completions` passthroughを使う。V4.1単体は`deepseek_v41` tokenizer／reasoning parser／tool parserによるnative `/chat/completions`を使う。モデル名だけを置換せず、後者を基準に接続する。
- 元会話のrole、本文、tool call/resultとID、reasoning履歴、画像の順序・境界を保持する。独立候補には、単体と同じ構造の元入力を直接渡す。
- V4.1は原画像を読む。旧`image_description`はtext-only V4への説明代替なので、native画像接続の成立後に依存関係ごと置き換える。説明文を原画像の代用にしない。
- 公開画像契約は既存ensembleのQwen側の制限（1画像、8 MiB、2,097,152 pixels）を維持する。V4.1単体の8画像対応を、そのままensemble全体の対応範囲としない。
- V4.1の基準samplingは単体exampleの`temperature=1.0, top_p=1.0`、effortは`low/high/max = 50/75/100`。旧V4の`top_p=0.95`や自由文のeffort preambleをV4.1の検証済み設定として引き継がない。
- 5ルートの`deepseek_direct`にはnativeの非thinking指定が必要。mainではeffort省略だけではV4.1のdefault thinkingを止められず、旧`reasoning_closed: true`も代用にならない。既存設定／transport拡張点で指定・優先順位を成立させ、実際のrendered inputで確認する。

## 4. Requirementとアンサンブル工程

### Requirementの継承仕様

ユーザーの補足に従い、closed PR #595の最終仕様（`31f1adc`の`requirements`）を基準にする。
担当はQwenではなくDeepSeek V4.1。元入力から、回答・次の行動が満たす条件を抽出し、課題そのものを解いたり回答を書いたりしない。

- 出力はMarkdown fenceや説明を付けない、空でないJSON配列。
- 各objectは文字列型の5項目のみ: `id`, `priority`, `requirement`, `acceptance_criterion`, `source`。
- IDは`R1`, `R2`, …の連番。`priority`は`minimum`または`optional`。
- 内容、範囲、形式、言語、除外条件、根拠、明示された手法・監査を抽出する。明示条件を勝手にoptionalへ降格しない。
- 数値・不等号・要求された完全一致文字列・句読点・改行を条件に保持し、JSONとしてescapeする。`source`だけに記載して条件本体を「指定どおり」で済ませない。
- 重複を整理しても独立に評価できる制約を落とさない。曖昧さを記録し、要求にない検証・制約や利用できない証拠を新たに要求しない。
- 元会話の指示の優先順位と後の訂正を尊重する。引用、画像内文字、tool結果、役割用scaffoldを新しい指示として扱わない。
- Checklistは派生データであり、元入力を置き換えない。後段は欠落・誤抽出も元入力から確認し、妥当なIDと条件を修正中も保持する。

Requirementのeffortは、未指定／low／highならnative high、明示max（L3でmaxへ正規化されるaliasを含む）ならnative maxとする。トップレベルのeffortを基準とし、矛盾するnested template値で変更しない。
他のDeepSeek thinking rolesはdefault high＋明示effort継承を基準とする。

mainのDSLは固定effortか`inherit`のみで、high下限とmax継承の組合せを直接表現できない。まずexample所有の設定・templateと既存request option経路で実現可能か確認する。成立しない場合は第7節の共有request契約として根拠を示すまで実装方法を未確定とし、固定highでmaxを消す代替は採用しない。旧helperや独自middlewareは復元しない。

### primaryの工程と入力

| role／工程 | モデル | 必須入力と出力 |
|---|---|---|
| `head` | Qwen非thinking | 元入力から既存の公開冒頭を生成、MAX256。依存なし |
| `requirements` | DeepSeek | 元入力のみから上記Checklist。候補を読まない |
| `policies` | DeepSeek | 元入力＋Requirementから異なる4方針 |
| `answer_1..4` | Qwen medium | 元入力＋Requirement＋全方針。それぞれ指定方針に従う独立候補 |
| `deepseek_candidate` | DeepSeek | 元会話・tools・画像を単体と同じ構造で読み、独立候補を生成。Requirement・方針・他候補を入力しない |
| `review`（批判的検討） | DeepSeek | 元入力＋Requirement＋全5候補から、主張・前提・根拠・成立条件を比較し指摘を残す |
| `synthesis`（再構成） | DeepSeek | 元入力＋Requirement＋全5候補＋review＋公開済みheadから回答を組み立てる |
| `audit`（独立監査） | DeepSeek | 別呼び出しで元入力・Requirement・候補・reviewとhead＋回答を照合。既存verifierとして`synthesis`を監査する |

旧`draft → critique`を独立DeepSeek候補へ置き換え、5候補の批判的検討を`review`として分離する。再構成と監査も別呼び出しにする。独立とは入力と呼び出しの分離であり、モデル間の誤りが統計的に独立という意味ではない。

構成には既存`roles`、`depends_on`、`verifies`、`prompt_headless`、role samplingを使う。verifierの追加依存は監査対象の直接依存にも必要なので、`synthesis`からRequirement・各候補・reviewへ明示的に依存させる。
既存Conductorはwave単位の実行であり、同じDeepSeek GPU群の複数roleは資源を共有する。新schedulerを作らず、実際のwave待ちと直列経路を測定する。

### 批判・再構成・監査の内容

1. 全5候補の結論だけでなく、前提・根拠・解法・成立条件を突き合わせる。多数派やDeepSeek候補を正解扱いしない。
2. 重要な主張について反例、境界条件、根拠不足、見落としを確認する。全候補共通の前提も対象にし、不要な反論は作らない。指摘には元入力・候補の参照位置と理由を残す。
3. 再構成では批判自体の妥当性も確かめ、重要な指摘の採用・不採用と理由を内部記録に残す。新しい結論は根拠・成立条件を明記し、仮説を確認済みの事実にしない。
4. 監査は既存の先頭`PASS`／`FAIL`に続けて、各Requirement IDの状態・証拠・修正指示を示す。妥当なminimumの未達／確認不能はFAIL、optionalの改善余地だけではFAILにしない。要求の脱落、指摘の放置、矛盾、飛躍、回答／tool-call形式を確認する。
5. FAIL指摘はそのまま修正へ渡し、最大2回の修正とその都度の独立監査を行う。修正後もFAILなら結果を改ざんせず、既存の最終試行公開方針を維持する。

mainには、判定不能時に1回再監査し、なお判定不能なら現試行を採用する分岐もある。これはFAIL修正上限とは区別して記録し、PASSに読み替えない。通常検証で判定不能や空監査しか出ない予算を合格にしない。

### 公開契約と維持する設定

- `primary`, `qwen_direct`, `qwen_think_medium`, `deepseek_direct`, `deepseek_think`の5ルートとjudge fallbackを維持する。Requirementはprimaryで実行し、通常の意味に基づく直答ルート選択は維持する。
- judgeはQwen非thinking、MAX8、timeout 5秒を維持する。headはjudge完了・primary選択後に早期配信される。リクエスト時刻からのTTFTにはjudgeを含める。
- 元入力は全量保持する。既存judgeの最新user最大4000文字の分類用viewは、後段が全文を読むことの代用にしない。容量対策としてviewを流用したり、judgeに直答を選ばせたりしない。
- Qwen mediumはmodel／revision、spec上の`reasoning_effort: high`によるmedium対応、`temperature=1.0, top_p=0.95, top_k=20`、profile MAX131072を維持する。現在使う`qwen3.8-chat.jinja`も変更しない。候補のMAX4096は十分性を再評価し、mediumやsamplingを落として不足を隠さない。
- `public_output_floor: 256`を維持する。tool、形式指定、`n>1`、`best_of>1`、logprobs等では既存どおりheadを無効にし、完全な回答／tool callを返す。
- 内部候補・review・監査を公開回答の代用にしない。公開済みheadを撤回・重複配信せず、監査はheadと本文を一つの回答として扱う。既存の内部出力表示とpublic contentの区別も維持する。
- mainの`n>1`監査省略は変更対象。choiceごとに監査・最大2修正・finish reason・usage・indexを保持し、監査前の本文を配信しない。`n>1`ではthinking後のfloor継続も無効になっているため、choice別の継続状態・予算・検証を併せて成立させる。
- 小さな公開MAXをheadが使い切るとmainはfinalと監査を省略し得る。公開上限を守りながら内部工程を完遂する予算・head条件を成立させ、この省略を合格扱いしない。
- 公開API、Chat UI、`kairyu-auto-max`、`embed-small`とembeddingの既存契約、`max_concurrency: 256`を維持する。

## 5. MAX調整と400・502の修正

### 障害の事実と再現

Issue #599では、Qwen直答がMAX131072を要求し、上流が入力を「少なくとも131073 token」、contextを262144と報告して400を返した。131073は完全なrendered inputの確定値ではない。原因がConductor／Orchestratorで失われ、`orchestration final unit produced no public output`という502となった。

当時のclientには明示MAXがなかった。mainにも会話全体のJSON文字列化と最新userの重複、入力残量に基づかないMAX、失敗を空出力扱いする経路がある。保存要求を使い、現在のnative単体入力とAUTO入力の差、rendered input拡大量を計測してから修正する。

### 送信する各呼び出しの容量

生成時と同じmodel、tokenizer／template、会話、tools、画像処理、role依存入力、assistant継続から実入力を確定する。すべてのdispatchで次を満たす。

```text
C = 実際のbackend context上限
I = そのdispatchで生成に使うrendered input token数
B = role／profileと公開・内部の区別から決まる出力上限
送信MAX = min(B, C - I)   （I < C、かつ生成を完遂できる予算が必要）
```

候補、policies、review、synthesis、audit、修正、再監査、thinking後の継続に同じ条件を適用する。依存出力がまだ存在しない受付時の見積もりだけで完了にせず、各呼び出し直前に確定する。
Qwen mediumのprofile MAX131072は設定上の上限として維持し、wire上のMAXだけを残容量以内へ調整する。

mainの`count_prompt_tokens_async`は`{model, prompt}`のbest-effort計測で、messages／tools／画像／assistant prefill込みの生成payloadとの一致を保証しない。既存API名があることを根拠に計測済みとはしない。計測できない入力を推測値で上流へ送って400を待つ方式も採用しない。

### 公開予算と内部推論の分離

- `OrchestrationRequest.internal_sampling_params()`は現在、公開MAXと内部capのminを取る。この結合を共有契約として修正する。
- 内部roleはrole別・model別の推論設定で予算を持つ。公開MAXをRequirement、候補、review、audit等へ流用しない。最終publisherに適用するcaller上限と、内部usage・資源会計は別に管理する。
- MAXのAPI上の意味は維持する。最終生成の上限にはその呼び出しのprivate reasoningも含み、headと公開継続は同じ公開予算を共有する。可視contentだけの上限へ変更しない。`n`はchoiceごとの上限を増減させず、usageは全内部呼び出し・監査・修正・継続の消費を計上するため、合計usageが公開MAXを超えることとは区別する。
- `internal_max_tokens: 65536`とUIの暗黙MAX65536を据え置いて解決したことにしない。UIで未指定と利用者の明示値を区別し、明示された公開MAXは守る。
- mainのDSLには`internal_max_tokens <= 131072`の制約もある。より大きな内部値が必要な場合は、role値だけを変えず、共有上限・予約・usageの契約を調べる。
- 旧V4の出力cap393216やV4.1単体UIの既定値を、V4.1内部推論の適正値として扱わない。単体の設定と実測を基準に、生成量、終了理由、打ち切り、候補・監査の完遂、遅延を測って調整する。
- thinkingのみで終了する候補、空の判定、途中で切れた必須出力を正常完了としない。性能を通すためのeffort低下や工程削除は行わない。
- `max_steps`等は新DAGとchoice数から算定する。通常の10生成role＋audit、headless差分、最大2修正、判定不能の再監査、空出力継続、容量処理を数え、既存の19を無条件に流用しない。

### エラーの保持

容量超過payloadを送らないことが本体の対策であり、400を返すだけでは完了にしない。
併せて、既存`UpstreamClientError`が持つ`status_code`・`code`・安全な`public_message`をConductor、Orchestrator、L3まで保持する。再試行可否はその原因から一貫して分類し、必要なら最小の明示的な伝播契約を加える。mainの例外型にretryability専用fieldがあるとは扱わない。利用者向けには安全な説明を返し、生のupstream URLや内部情報を露出しない。

mainのReplicaPoolは4xxとhealth障害を区別しているため、その契約を維持する。非retryableな入力エラーを空出力として再dispatchしたり、原因不明の502へ変換したりしない。
すでにstreamを開始した場合はHTTP statusを変更できないため、SSE上のエラー・trace・usage・終了処理まで確認する。失敗した必須roleを空文字で後段へ渡し、headだけや不完全な統合を成功扱いする経路も確認する。

## 6. 入力自体が容量を超える場合――未解決の設計条件

MAX補正は`I >= C`や、残量では必須出力を完遂できない場合を解決しない。原文、全候補、検討結果、回答をリクエスト単位で全文保持し、容量内の範囲の読み取り・再読・編集を有限の仕事として管理する要件は維持する。画像も原画像と領域位置を保持する。

ただしmainのDSLは固定DAGと文字列templateであり、一般的な範囲store、動的な再読・編集、必須範囲の処理記録は存在しない。MoAは候補全文を連結し、Conductorの修正は前回答と指摘をpromptへ追加する。既存executorはcode/test sandboxであり、この文書処理を提供するものではない。

従って、旧計画の「既存経路へ接続すれば実現できる」という前提を撤回し、次を解くまでこの部分の実装方式は未確定とする。

1. 実在するMoAの長い候補連結とConductorの修正履歴で、独立した容量破綻を再現する。
2. 既存拡張点でどこまで構成でき、何が不足するかをコード経路で特定する。
3. frameworkに必要な最小の共有契約と、exampleがDSL／設定で指定する探索・編集方針を分離する。参照・容量内処理・出力保持・完了記録の必要性は候補であり、これら一式の採用を事前承認したとは扱わない。
4. 必須範囲の受け渡しと実dispatchを、source ID・位置・版・入力の対応・実行結果で検証できることを示す。データを保存したこと、モデルが「全部読んだ」と答えたことだけでは処理完了としない。
5. 全5候補の比較と相互参照、監査・修正が有限に完遂し、usage・stream・取消の既存経路へ接続できることを示す。

無制限の入力を有限contextだけで処理できると保証せず、入力上限と有限作業の成立条件を明示する。ただし既存で受理する要求を新たな小さな上限で拒否したり、静的な先頭数ページだけを処理したりして本要件の達成とはしない。
汎用REPL、永続DB、別scheduler、別HTTP基盤、role名で動作を切り替えるframework文書runnerは追加しない。

この条件を解決できない限り、短い入力のアンサンブルが動いても全体を実装完了としない。

## 7. framework修正の採用判定

境界ruleの4点――共有契約とコード経路、既存拡張点では不足する理由、対象example以外の具体的利用と回帰、最小機構とexampleに残る方針――を、変更前に記録する。以下は確認した欠落と修正を絞る対象であり、旧branchの差分一式を採用する許可ではない。

| 共有契約とmainの経路 | 既存設定では足りない点／独立した回帰 | 修正を絞る境界 |
|---|---|---|
| 構造付き会話・request metadata: `chat_service.validate_orchestration_chat_input` → `OrchestrationRequest` → `Conductor._request_intent` → `OpenAIBackend._payload` | text履歴がJSON文字列になり、最新userが重複。内部roleにはtoolsが渡らず、画像の派生promptも元のmessage配置を失う。通常のAUTO直答・tool会話でも再現できる | 既存request／generation carrierとtransportを必要最小限拡張。独立候補のrole名やRequirement方針を埋め込まない |
| 実入力に基づくMAXと公開／内部予算: `OpenAIBackend.count_prompt_tokens_async`、`OrchestrationRequest.internal_sampling_params`、Conductorの各dispatch | 現行計測は実生成payloadを表さず、role overrideも公開capとの結合を外せない。通常の単一role直答、既存MoA／verifierでも長文400・内部予算不足が起きる | 同じ生成入力からの容量確定と既存予約／usageへの接続。モデル別数値・工程の予算配分は設定へ置く |
| 失敗原因と依存実行: `Conductor._generate`／`_run_unit_safe` → `Orchestrator.run` → L3 error変換 | 現行設定で消えた例外原因を復元できない。単一publisherの4xxや、依存入力の欠落を正常出力と区別できないことが独立した回帰になる | 既存エラー・実行結果・依存の欠落状態を最小限伝播する。必須工程と公開成功の方針はexampleのDSL／設定に残す |
| 回答ごとのverifier: `Conductor._run_unit`／`stream`とchoice出力 | `n>1`でコードが明示的に監査とfloor継続を省略し、設定では有効化できない。既存の任意verifier付きpublisherにも同じ回帰がある | choice単位の監査・修正・公開待ち・継続状態・予算・usage・取消。判定prompt、FAIL基準、最大2回という方針はexampleへ置く |

mainのbest-so-farは意図された共有動作（O4／EO-D7）でもある。原因保持の修正を理由に全Conductorのfallbackを一律禁止しない。このexampleの「必須工程が欠けたら公開成功にしない」方針を既存DSLで表現できない場合は、独立した利用・回帰と最小拡張の根拠を示すまで方式を未確定とし、任意の追加switchを自動的に採用しない。

Requirementのeffort下限やV4.1非thinking指定は、まず既存template／request optionの構成を試す。不足が共有metadata契約の問題である場合にだけ上記基準で評価する。`requirements`というroleを検出してhighへ変更するような処理はframeworkに置かない。
容量超過時の範囲処理は第6節の未解決条件であり、「MoAでも長くなる」という類似だけでframework採用範囲へ含めない。承認済み共有契約を超える拡張が必要なら、具体的な差分と根拠を用意してから別途扱う。

## 8. 検証と性能要件

### 実行・回帰の検証

- 保存された#599の要求、caller MAX未指定／明示、native単体で受理できる境界会話を使う。上流へ届いた実payloadとMAXを確認し、原文・tool構造・画像境界・完全一致文字列を保つ。
- 全候補と各工程の入力・出力対応、RequirementのJSON／IDと後段伝播、high下限／max継承を検証する。モデルの自己申告だけで工程完了を判定しない。
- 長い候補・review・修正履歴、thinking後の継続、空出力、非retryableな4xx、後段失敗を通す。容量内調整と第6節の範囲処理を別に検証する。
- 監査のPASS、FAIL→修正、2回修正後FAIL、判定不能→再監査を区別する。複数choiceでは異なる判定と修正を与え、choice index、tool call、公開順、usageが混線しないことを確認する。
- stream／非stream、headless、response format、logprobs、取消、監査中の切断、後続要求での資源解放、Chat UI、embeddingを確認する。
- 既存`verification.py`のstage一覧、trace検証、runtime照合を更新する。失敗roleを無視した200や、headのみの回答を全工程成功と扱わない。

テストは既存のrequest、Conductor/head、OpenAI backend、ReplicaPool、serverのusage/trace、example運用・検証のbehaviorテストを優先する。実際の受理入力から公開結果・backend要求までの回帰を守り、設定一覧を再列挙するためのテストや新しいexample専用Python helperは追加しない。
`n>1`監査省略を期待するテストは新契約に更新する。旧stageの削除に伴う専用test/helperは削除し、削除作業ではCLAUDE.mdどおり同条件でbase/headのcollection数と残す領域の理由を報告する。

### 性能の合格条件

| 項目 | 条件 |
|---|---|
| Semantic TTFT | 同じ入力・同じ並列数・対応するeffortの新6 GPU DeepSeek L1単体に対し、最初の公開contentまでのp50が2倍以内 |
| 負荷 | c1／c8／c16／c32、各32要求。自然なjudge選択とprimary必須の行を両方測る |
| 同時受付 | `max_concurrency: 256`を維持 |
| 工程 | primaryの全5候補、Requirement、policies、review、synthesis、監査、必要な修正を実際に実行する |
| 正常応答 | 非空の公開回答、または要求に対応した正しいtool call、正しいusage・trace・終了理由 |
| エージェント利用 | 既存の900秒ターン予算で完遂・tool call・監査を確認する |
| 公開順序 | 許可された要求でQwen headを早期配信し、本文は該当choiceの監査後に配信する |

現在のharnessは、全要求がthinking直答を選んだ行を`not_applicable`として合格でき、paired baseline失敗時に過去V4の値へfallbackする箇所もある。新構成のprimary合格判定では両方を認めない。
primary必須検証は既存harness／検証用設定から同じprimary DAG・予算を通し、各要求のroute・必須stage・監査をtraceで確認する。同じL3受付と実際のMAX8 judge呼び出しを直列経路に残し、その遅延込みで測る。primaryへの強制方法と設定差分も証跡に残す。judgeを除いたDAG単体の時間は診断値とし、合格根拠にしない。自然なjudge選択の測定は別に残す。

比較用L1は新しい6 GPUの稼働設定で新規測定する。V4の過去値、#598の旧実装値、V4.1 TP8の値を分母に使わない。V4.1単体の固定256-token行はreasoning込みのmodel TTFTであり、公開contentのない行をSemantic TTFTの分母に使わない。
同じdataset・並列数・対応effortで各32要求が正常完了し、非空の回答を返すpaired baselineを用意する。旧harnessの直接比較MAX512を流用せず、単体の推論と回答を完遂できるMAXを設定する。欠測、失敗、length打ち切りの行を分母として採用しない。
headlessのtool／形式指定等も全工程と完遂を確認し、headありTTFTの合格証拠と混同しない。

E2E、TPOT、スループット、TTFT・E2Eのp99、成功率、route別TTFT、judge遅延、2レプリカへの配置、工程別生成量・終了理由・待ち時間も記録する。900秒を全長文要求の一律E2E上限にはしない。性能を満たすために必須工程や思考量を削らない。

## 9. 実装順序・成果物・PR

1. **設計を具体化する。** 本書のmain基準を再確認し、6 GPUの並列成立性、native構造入力、Requirement effort、V4.1非thinking、容量超過処理、choice別監査を解決する。採用不可のframework拡張を含めず、未解決条件を隠さない。
2. **exampleを設定で構成する。** 計画承認後、最新mainから`codex/` prefixの新branchを作る。新exampleに`compose.yaml`, `kairyu.yaml`, `auto-max.yaml`, `example.json`と必要なshell入口を作成する。既存exampleのファイルは変更しない。V4.1 runtime資産はmainを基準に再利用する。
3. **採用条件を満たす共有修正を行う。** 原因ごとに小さく実装し、対象example以外の既存経路で回帰を確認する。main-to-PR全差分で所有権を確認する。
4. **検証可能な単位でcommit・pushする。** 新しいdraft PRへ構成、変更理由、framework契約の根拠、未検証項目を記載する。#600はmerged、#595／#598はclosedの状態を維持する。
5. **8 GPU環境へ配置する。** 現稼働の設定・stateと復帰方法を記録し、pull後にL1全3サービス、Kairyu L2/L3、UIを更新構成で再起動する。通常の入口・readinessから起動できることを確認する。
6. **稼働物を照合する。** ローカルhashの記録に加え、全3 L1サービスの実Docker Image／Cmd／Env、古いopt-in設定の残存、GPU割当、稼働revisionとcheckpoint manifest、全poolのbackendを照合する。mounted L2設定・template・rendering処理とUI filterもcheckoutに一致することを確認し、不一致なら測定を開始しない。その後に機能・容量・取消検証を行う。
7. **性能とagent利用を検証する。** fresh native baseline、自然routing、primary必須matrix、900秒条件を実行する。必要なチェックが通った後の追加・再実行は、変更や失敗の影響に応じて行う。
8. **証拠を保存する。** 新しい構成と結果を新exampleの`MEASUREMENTS.md`へ記録し、既存exampleの測定記録を変更しない。README、設計文書、PRを最終実装に合わせ、設計決定・障害・進捗は規則に従って`PROGRESS.md`へ記録する。

容量起因の400、誤った502、必須工程の省略、内部推論の公開MAXによる不当な制限、未解決の容量超過処理、未検証の6 GPU構成または性能が残る状態を実装完了としない。監査が常に正しいという保証や113タスクの再実行は追加しない。

## 10. 確認元

- [main基準commit](https://github.com/ytworks/kairyu/commit/99c5eadbc67d32d57823accd89aac34215e2c720)、[PR #600](https://github.com/ytworks/kairyu/pull/600)。
- [旧RequirementのPR #595](https://github.com/ytworks/kairyu/pull/595)、[最終仕様のrequirements role](https://github.com/ytworks/kairyu/blob/31f1adc1caaacb7b8689a811aa65ea7ff4b74ffb/examples/qwen3.8-deepseek-v4-8gpu/auto-max.yaml#L191)。仕様を参照し、旧実装と測定値は採用しない。
- [closed PR #598](https://github.com/ytworks/kairyu/pull/598)、[Issue #599](https://github.com/ytworks/kairyu/issues/599)。
- mainの[既存ensemble設定](../../../examples/qwen3.8-deepseek-v4-8gpu/auto-max.yaml)、[測定記録](../../../examples/qwen3.8-deepseek-v4-8gpu/MEASUREMENTS.md)、[検証入口](../../../examples/qwen3.8-deepseek-v4-8gpu/verification.py)。
- mainの[V4.1単体設定](../../../examples/deepseek-v4.1-flash-8gpu/example.json)、[測定記録](../../../examples/deepseek-v4.1-flash-8gpu/MEASUREMENTS.md)。
- mainの[DSL](../../../kairyu/dsl/spec.py)、[request](../../../kairyu/orchestration/request.py)、[Conductor](../../../kairyu/orchestration/conductor.py)、[Orchestrator](../../../kairyu/orchestration/orchestrator.py)、[chat入力／エラー](../../../kairyu/entrypoints/server/chat_service.py)、[OpenAI backend](../../../kairyu/engine/openai_backend.py)、[画像prompt](../../../kairyu/engine/prompt.py)、[MoA](../../../kairyu/orchestration/moa.py)。
