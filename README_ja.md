<!-- This file is the Japanese translation of README.md. When you change either README, update the other so both languages stay in sync. -->

# HOOPS AI 類似検索サーバー

**言語 / Languages:** [English](README.md) | 日本語

## このリポジトリの位置づけ

![このリポジトリの位置づけ: 下層に Python の機械学習フレームワークである HOOPS AI が
あります。デスクトップアプリはネイティブブリッジ経由でそれを呼びます。このリポジトリは
もう一方の経路を取り、FastAPI のサービスが HOOPS AI を包み、HOOPS Visualize Web で
作ったブラウザクライアントが HTTP で話します。破線の枠がこのリポジトリの範囲 ―
Web App、Web API、その背後の HOOPS AI 呼び出しです。AI エージェントは MCP 経由で同じ
エンドポイントに到達します。](images/how_to_use_hoops_ai.png)

HOOPS AI は Python の機械学習フレームワークであり、それをユーザーに届ける経路は1つでは
ありません。

デスクトップアプリは、ネイティブブリッジ経由で HOOPS AI を呼びます。このリポジトリが
取るのは Web の経路です。FastAPI のサービスが HOOPS AI を包み、HOOPS Visualize Web で
作ったブラウザクライアントが HTTP でそのサービスと話します。図の破線の枠がこのリポジトリ
の範囲です ― Web App、Web API、そしてその背後の HOOPS AI 呼び出し。

サーバーが素の HTTP を話すため、クライアントはブラウザのものだけに限りません。AI
エージェントが MCP 経由で同じエンドポイントを叩くこともできますし、リクエストを送れる
ものであれば何でも構いません。

## ステータス: サンプル / 概念実証

このリポジトリは **サンプルであり概念実証（PoC）** です。HOOPS AI の類似検索を、
HOOPS Visualize Web をフロントエンドに持つオンプレミス型サーバーとしてどのように組み立て
られるかを示すために存在します。Tech Soft 3D のパートナーが、こうしたシステムの全体像を
把握し、そこからアイデアを得られるようにすることが目的です。

**これは製品ではなく、そのまま出荷できるものでもありません。** このコードは製品品質の
ソフトウェアではなく、あくまで出発点として扱ってください。無保証であり、サポートの
確約もありません。

### これをベースに配布を検討する場合

ここで使用している HOOPS SDK コンポーネント（HOOPS AI、HOOPS Exchange、
HOOPS Visualize Web（そのオーサリングライブラリである libsc / hps_core を含む）、
および学習済みチェックポイント）は、Technology Partner Agreement とお客様のライセンスに適用される
その他の条項によって規定されます。**これらを再配布できるか、またどのような形で
再配布できるかは、お客様ご自身の契約に依存します。** このリポジトリを答えとみなさず、
お客様に適用される条項をご確認ください。

統合製品を配布する際に伴う義務についても同様です。再掲すべき proprietary notice、
製品表示、エンドユーザーとお客様自身の契約条項などは、お客様側で対応いただくもので
あり、このリポジトリが代行するものではありません。

### パッケージングスクリプトと生成されるライセンス文書もサンプルです

`tools/build_redist_package.ps1` / `.sh` と、それらが生成するファイル
（`THIRD-PARTY-NOTICES.md`、`third_party_notices/`、`THIRD_PARTY_MANIFEST.txt`）は、
自己完結型パッケージを組み立て、同梱した各コンポーネントの出所を記録する *ひとつの方法*
を示すものです。

**これらはプロセスの例示であって、コンプライアンス上の成果物ではありません。** これらは
特定の依存関係セットに対し、特定のマシン上で、特定の時点で生成されたものです。法務による
レビューは受けていません。HOOPS Visualize Web の2つの謝辞テキスト（ブラウザビューアと
その WebAssembly エンジンのもの、および Stream Cache ライブラリ `libsc` のもの）は、
バージョン表記のない公開
ドキュメントサイトから取得したものです（`third_party_notices/README.md` を参照）。
製品を出荷する場合は、ご自身のビルド向けに notice を生成し、検証してください。

[HOOPS AI](https://www.techsoft3d.com/developers/products/hoops-ai/)（Tech Soft 3D）
の上に構築された、**CAD の類似検索および類似比較に特化した** FastAPI ベースの REST
サーバーです。

このサーバーが行うことは1つ、CAD の部品やアセンブリを形状で検索・比較することです。
形状検索、形状同士の比較、一括登録を備えた名前付きインデックス管理、タグ、ギャラリー
表示用の部品ごとのサムネイル／ストリームキャッシュ（SCS）を提供します。想定読者は、
この設計をもとに自社製品を作る ISV です。ここにあるコードはリファレンスであって
そのまま出荷するものではありませんが、これが描くシステムは ISV がエンドユーザーへ
再配布する類のものです。

**2つの性格があります。** もともとは、HOOPS AI を自社製品に組み込む ISV パートナー
向けの実用サンプルでした（REST Web API、1 台のサーバーに対する複数のクライアント、
ライセンスキーの難読化、再頒布パッケージを作るためのツール）。そこに、別の問いを
扱う実験的な Web アプリという性格が加わりました。埋め込みベクトルをどうクラスタ化
すれば部品の種類に分かれるのか、という問いです。Shape Map、k 平均法、Outliers、
Drill in、Ward ツリー、パーツカタログはいずれもそこから来ています。どちらの目的で
読んでも構いません。設計から着想を得るために読んでも、クラスタリングそのものの
実験台として読んでも結構です。

---

## アーキテクチャ

このサーバーは FastAPI アプリで、その HTTP ハンドラはほとんどロジックを持ちません。
各トップレベルの関心事は、それぞれ単一目的のモジュールに分かれています。`core.py` は
薄い **ファサード** です。以下の16個のドメインモジュールを import し、それらの安定した
名前を再輸出するため、ルーターやツールはすべてを `core.<name>` として参照できます。

| モジュール | 責務 | 行数 |
|---|---|---|
| `config.py` | サーバー設定。データルート、その派生パス、`.env` ヘルパー | 225 |
| `locks.py` | プロセス全体のロック階層と、その順序規則を集中管理するレジストリ | 143 |
| `storage.py` | CAD ファイルの保存と取り込み。アップロードストア、ZIP 展開、ルックアップ | 509 |
| `embeddings.py` | 形状の埋め込み。エンコーダ、2段構成の埋め込みキャッシュ、モデル解決 | 709 |
| `index_registry.py` | 名前付きインデックスのレジストリ。増分インデックス管理と公開インデックス API | 1593 |
| `search.py` | 部品検索のエントリポイントと、その完成応答キャッシュ | 372 |
| `context.py` | Context Layer。近傍から生産レコードを予測する | 359 |
| `assembly.py` | vendored な AssemblyMatcher を用いたアセンブリ同士の検索 | 606 |
| `shape_map.py` | インデックスの Shape Map。2D クラスタマップと点群 SCS のエクスポート | 1256 |
| `tags.py` | インデックスのタグサイドカー。インデックスごとの `tags.json`、アプリケーション全体のタグ語彙、タグのキャッシュ、タグフィルタ API | 978 |
| `viewer.py` | HOOPS Visualize Web 向けの SCS ストリームキャッシュのエクスポート | 59 |
| `licensing.py` | 起動時に HOOPS AI ライセンスを有効化する | 157 |
| `warmup.py` | 名前付きインデックスに対するオプトインの起動時ウォームアップ | 199 |
| `jobs.py` | バックグラウンドのインデックス登録ジョブ | 379 |
| `runtime.py` | ランタイムディレクトリのライフサイクルと起動時の設定ログ出力 | 214 |
| `diagnostics.py` | リクエストごとのフェーズ計測。検索・埋め込みからは独立 | 49 |

**依存の向き。** 依存は厳密に一方向です。

```
main.py  ->  routers/  ->  core.py (facade)  ->  domain modules  ->  config / locks / storage
```

最下位のモジュールが上位のモジュールを import することはなく、**どのモジュールも
`core` を import しません**。

既定では、ランタイムデータは `<app>/data/`（データルート）配下に置かれ、ソースツリーと
生成される `uploads/ indexes/ out/ …` ディレクトリを分離します。`HOOPS_AI_DATA_DIR`
でその場所を上書きできます。

**なぜ `core.py` に3つの関数が残っているのか。** `remove_from_index`、
`list_index_parts`、`delete_index` は、それぞれ兄弟ドメイン（index + tags + context
+ assembly）にまたがる処理をオーケストレーションします。これらの兄弟は、いずれかの
インデックス所有モジュールより *上位* に位置するため、オーケストレータをどれか1つの
モジュールに押し込むと上向きの呼び出しが生じ、循環が発生します。3つをファサードに
残しておくことが、依存グラフを一方向に保つ理由です。

このレイアウトは実行時に強制されます。`tests/test_module_layout.py` は全モジュールを
走査し、いずれかが `core` を import した場合、再輸出が所有モジュールからずれた場合、
またはドメインモジュールが import 済みのモジュール名を隠した場合に失敗します。

境界が *なぜ* このような形になっているのか（再輸出の規約、キャッシュ無効化の契約、
ガードが守っている落とし穴）については、
[docs/architecture.md](docs/architecture.md) を参照してください。

---

## 要件

- Python 3.12
- 有効な **HOOPS AI ライセンスキーおよび HOOPS Visualize Web ライセンスキー** —
  後者は、Web ビューア向けのストリームキャッシュのオーサリングと、Shape Map の
  点群ツールのビルドに必要です
- 環境にインストールされた HOOPS AI（CPU 版または GPU 版）
- **HOOPS AI Tutorial Data Assets** — CAD 形状を埋め込むには、事前学習済みの
  埋め込みモデルチェックポイントが必要です。
  [HOOPS AI evaluation guide](https://docs.techsoft3d.com/hoops/ai/getting_started/evaluate.html#tutorial-data-assets)
  に従い、Tech Soft 3D の File Transfer サービスからデータアセットを入手してください。
  必須のファイルはチェックポイント
  `packages/trained_ml_models/ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt` です。

  > [HOOPS-AI-tutorials](https://github.com/techsoft3d/HOOPS-AI-tutorials/tree/main)
  > GitHub リポジトリには、チュートリアルの *ノートブック*（サンプルコード）のみが
  > 含まれます。このサーバーはそれらのノートブックには **依存しません** — File Transfer
  > で別途配布されるチュートリアルの **データアセット**（モデルチェックポイント）に
  > 依存します。

  **ディレクトリ構成** — `packages/` は `HOOPS_AI_SDK_DIR` の直下に置く必要があります。

  ```
  <HOOPS_AI_SDK_DIR>/
  └── packages/
      └── trained_ml_models/
          └── ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt
  ```

  **自分の CAD をインデックス化するのが主なワークフローです。** チェックポイントを配置
  すれば、エンドユーザーは自分の CAD ファイルから自分のインデックスを構築できます
  （アップロード + `/similarity/index/*`。[一括登録ジョブ](#一括登録ジョブ)
  を参照）。

---

## セットアップ

### 1. 依存関係のインストール

HOOPS AI（CPU 版または GPU 版）は、お使いの HOOPS AI ディストリビューションの手順に
従って別途インストールしてください。そのうえで、このサーバーの依存関係を
**HOOPS AI の仮想環境** にインストールします。

**Windows:**
```bat
<Path\to\HOOPS_AI\install\dir>\.venv\Scripts\pip.exe install -r requirements.txt
```

**Linux:**
```bash
/path/to/HOOPS_AI/install/dir/.venv/bin/pip install -r requirements.txt
```

> Ubuntu 22.04 以降ではシステムの Python が外部管理（PEP 668）となっており、素の
> `pip install` は拒否されます。HOOPS AI venv の pip を使えばこれを回避でき、サーバーを
> 実行するのと同じ Python に必要なパッケージがすべて入ることが保証されます。

#### ヘッドレス Linux (Ubuntu) の追加手順

ヘッドレスの Ubuntu サーバー（GPU なし、モニタなし）で HOOPS AI を実行する場合は、
手順を追った説明について次のコミュニティフォーラム投稿を参照してください。

**[I tried running HOOPS AI v1.1 headless on Ubuntu 24.04 EC2 — Tech Soft 3D Forum](https://forum.techsoft3d.com/t/i-tried-running-hoops-ai-v1-1-headless-on-ubuntu-24-04-ec2/5165)**

#### 任意: UMAP レイアウト

`umap-learn` をインストールすると、マップの **Layout** に PCA と並んで UMAP が
現れます。変わるのは点が描かれる位置だけで（クラスタリングはどちらの場合も 2,048
次元のベクトル上で実行されます）、`HOOPS_AI_SHAPE_MAP_UMAP_N_NEIGHBORS`、
`HOOPS_AI_SHAPE_MAP_UMAP_MIN_DIST`、`HOOPS_AI_SHAPE_MAP_UMAP_SEED` で調整します。
導入手順、導入前後に確認すべきこと、コスト、切り戻しの方法は
[docs/umap_ja.md](docs/umap_ja.md)（英語版: [docs/umap.md](docs/umap.md)）に
あります。

### テストの実行

テストスイートは SDK 不要で、HOOPS AI ライセンスも必要ありません。`requirements.txt`
に加えてテスト専用の依存関係をインストールし、pytest を実行してください。

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
```

GitHub Actions のワークフロー（`.github/workflows/tests.yml`）は、`main` への
すべての push とプルリクエストに対し、この同じ SDK 不要の pytest を Python 3.12 の
Windows ランナー上で実行します。Windows はこのサーバーのターゲットプラットフォームで
あり、スイートはそこで検証されています。HOOPS AI SDK もフロントエンドのパッケージも
インストールせず、シークレットも不要です。

### 2. 環境変数の設定

`.env.example` を `.env` にコピーし、値を記入してください。

```bash
# Windows
copy .env.example .env

# Linux
cp .env.example .env
```

| 変数 | 必須 | 説明 |
|---|---|---|
| `HOOPS_AI_LICENSE` | 開発用フォールバック | お使いの HOOPS AI ライセンスキー。**`.env` ファイルからのみ** 読み込まれ、システム環境変数からは決して読み込まれません。再配布パッケージでは、キーは代わりにコンパイルされた license shim から供給されます。この `.env` の値は開発時のみのフォールバックです（*License key handling* を参照）。 |
| `HOOPS_AI_SDK_DIR` | 任意 | HOOPS AI SDK インストールディレクトリへの絶対パス（`packages/` を含む必要があります）。フル SDK がインストールされたマシンで、そのインストールに同梱されたチェックポイントを再利用できます。未設定の場合（再配布パッケージ）、チェックポイントはデータルートとアプリディレクトリから解決され、名前付きインデックスは引き続き動作します。 |
| `HOOPS_AI_DATA_DIR` | 任意 | `uploads/ indexes/ jobs/ out/ embeddings_cache/ models/` の親ディレクトリ。未設定の場合は `<app>/data` が既定になります（[データディレクトリ](#データディレクトリ) を参照 — ランタイムディレクトリは `data/` 配下に置かれ、ソースとは分離されます）。別のボリュームにデータを置くには設定します。 |
| `HOOPS_AI_EMBEDDINGS_MODEL_NAME` | 任意 | 使用する埋め込みチェックポイント `*.ckpt` のファイル名（例: `ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt`）。設定した場合、`<DATA_ROOT>/models/`、次に `<APP_ROOT>/models/`、次に `<HOOPS_AI_SDK_DIR>/packages/trained_ml_models/` の順に探し、最初に見つかったものを使います（同じ場所にある他の `*.ckpt` は無視されます）。未設定の場合、`<DATA_ROOT>/models/`、次に `<APP_ROOT>/models/` から唯一の `*.ckpt` を解決します。 |
| `HOOPS_AI_INDEX_CACHE_MAX_MB` | 任意 | プロセス内 **名前付きインデックス書き込みストア**（create/add/remove で使用、検索では未使用）のメモリ予算。これを超えると最も長く使われていないインデックスが退避され、次の書き込み時にディスクから透過的に再ロードされます。既定は `1024`（`tmcad` サイズのインデックス約3個分）。[メモリサイジング](#メモリサイジング) を参照。 |
| `HOOPS_AI_EMBED_CACHE_MAX_MB` | 任意 | プロセス内 **埋め込みキャッシュ** のメモリ予算。ディスクにバックアップされているため、退避されたエントリは再計算ではなく再ロードされます。既定は `256`。 |
| `HOOPS_AI_WARMUP_INDEXES` | 任意 | 起動時にバックグラウンドでウォームするインデックス名（カンマ区切り。searcher、Shape Map、アセンブリマッチャ）。これにより最初のユーザーがコールドビルドのコストを負担しなくなります。**未設定はウォームアップなし** を意味します（既定）。[起動時ウォームアップ](#起動時ウォームアップ) を参照。 |
| `HOOPS_AI_WARMUP_MAX` | 任意 | 設定した `HOOPS_AI_WARMUP_INDEXES` のうち実際にウォームする数の上限。インデックスの多いホストでも素早く起動できます（既定 `1`）。 |
| `HOOPS_AI_WARMUP_STAGES` | 任意 | 実行するウォームアップ段階をカンマ区切りで指定します: `assembly`、`map`、`scs`、`searcher`。未設定の場合はすべて実行します。アセンブリマッチャが最も時間のかかる段階なので、アセンブリ検索を使わない構成では外せます。[起動時ウォームアップ](#起動時ウォームアップ) を参照。 |
| `HOOPS_AI_DEMO` | 任意 | Web クライアントのうち、ウェビナーのデモ専用の部分を表示します — 現在は Context Layer から生産レコードを予測する **Estimate** パネルです。設定しない限りオフです（`1`/`true`/`yes`/`on` で有効）。この予測は、インデックスに実際の生産レコードが付いて初めて意味を持ちますが、デモでは捏造したレコードをシードしているためです。コンテキストの API 自体は、どちらの場合も利用できます。 |

上の表には、ユーザーや運用者が通常設定するものだけを載せています。それ以外のつまみ
（アップロード/ZIP の上限、TTL、ジョブ上限、インデックスごとのキャッシュとプリウォームの
トグル、ワーカーのサイジング、埋め込みの intra-op スレッド、検索キャッシュや診断フラグなど）
は、いずれも妥当な既定値を持ち、1箇所にまとめてあります。`config.py` の先頭付近にある
**Environment variable reference** ブロックです。これらも同じように環境変数や `.env` で
設定できます。ここに重複して載せていないだけです。

`.env` の例:

```
HOOPS_AI_LICENSE=XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
HOOPS_AI_SDK_DIR=C:\hoops_ai
HOOPS_AI_EMBEDDINGS_MODEL_NAME=ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt
```

### ライセンスキーの扱い

このサーバーは ISV からエンドユーザーへ再配布されるため、ISV の HOOPS AI ライセンス
キーがエンドユーザーのディスク上に平文で置かれるべきではありません。キーは代わりに、
Nuitka でビルドしたコンパイル済み拡張モジュール（*license shim*）の内部に持たせることが
でき、設定ファイル・ログ行・プロセス環境のいずれにも現れないようにできます。

**起動時にキーがどのように読み込まれるか**

`core.init_hoops_license()` はまず `import hoops_license_shim` を試みます。この
コンパイル済みモジュールが存在する場合、その唯一の公開関数 `activate()` が内部で
キーを復元し、自身で `hoops_ai.set_license()` を呼び出します — キーの値がサーバー
自身のコード内で Python 文字列になることはありません。コンパイル済みモジュールが
存在しない場合、サーバーは `.env` ファイルからのみ `HOOPS_AI_LICENSE` を読み込む
フォールバックに切り替え（システム環境変数からは決して読み込みません）、警告を
ログに出します。このフォールバックは開発用です。配布デプロイでは常にコンパイル済み
shim を同梱すべきです。shim が有効でも `.env` に平文の `HOOPS_AI_LICENSE` がまだ
残っている場合、サーバーは未使用のキーを削除できるよう警告も出します。

**shim のビルド（オペレーターの手順）**

**サーバーを実行するのと同じ Python 3.12** で、SDK 仮想環境の中からビルドします。
手順はプラットフォームごとに異なります。

*Windows (PowerShell):*

```powershell
# SDK の仮想環境の中で
python -m pip install nuitka                 # 未インストールなら一度だけ
$env:HOOPS_AI_LICENSE_BUILD = '<your-key>'   # シングルクォート: PowerShell は中の $ を展開しません
python tools/build_license_shim.py           # --dry-run を付けると確認だけ行います
Remove-Item Env:\HOOPS_AI_LICENSE_BUILD
```

*Linux (bash):*

```bash
# Nuitka は C のツールチェーンを必要とします。一度だけ導入します:
sudo apt-get install -y build-essential patchelf

source /path/to/your/.venv/bin/activate      # SDK の仮想環境を有効化します
python -m pip install nuitka                  # 未インストールなら一度だけ
# 先頭の空白でキーをシェル履歴に残しません (HISTCONTROL=ignorespace が必要)
 export HOOPS_AI_LICENSE_BUILD='<your-key>'
python tools/build_license_shim.py            # --dry-run を付けると確認だけ行います
unset HOOPS_AI_LICENSE_BUILD
```

Ubuntu では解釈系は `python3`（または `python3.12`）です。`python` コマンドは
有効化した仮想環境の中でのみ存在します。ABI タグが一致するよう、サーバーを実行するのと
同じ解釈系で `.so` をビルドしてください。

ビルドスクリプトはキーを難読化し、中間ソースを生成し、Nuitka を呼び出し、`finally`
ブロックで中間ソースと Nuitka のビルドツリーを必ず削除します。成果物は
`hoops_license_shim.<abi-tag>.pyd`（Windows）または `.so`（Linux）です。import 可能に
なるよう `core.py` の隣に配置してください。

**なぜ shim はソースではなくリリース成果物なのか**

`.pyd`/`.so`、生成された中間ソース、Nuitka のビルドツリーはすべて `.gitignore`
対象です。Git 履歴からは、たとえ難読化されたものであっても、一度混入した埋め込み
キーを取り除くことはできないためです。コンパイル済み shim は、リリース作業の一環と
してビルドし、ソースツリーの中ではなくパッケージと並べて配布してください。

**Python バージョンとプラットフォームの結合**

コンパイル済み shim には ABI タグが付きます（例:
`hoops_license_shim.cp312-win_amd64.pyd`）。Python ランタイムをアップグレードすると
（例: 3.12 → 3.13）shim は import 不可になり、その時点でサーバーは静かに `.env`
フォールバックに戻ります（警告付き）。Python をアップグレードしたら shim を再ビルド
してください。Linux ディストリビューションはそれぞれ独自の `.so` ビルドが必要で、
Windows の `.pyd` はそこでは読み込めません。

**これが保護するもの、しないもの**

これは **難読化であって暗号化ではありません** — 難読化を解く鍵は同じバイナリの中に
同梱されます。ライセンスキーが設定ファイル・ログ出力・プロセス環境を通じて露出する
ことや、成果物の `strings` 出力に現れることを防ぎます。しかし、キーが渡される瞬間に
それを捕捉するために `hoops_ai.set_license` API を差し替えるような、本気の相手を
止めることは **できません**。これは shim が防げない Python レベルの傍受であり、
それ以上の保証は主張していません。

### 3. Web クライアントのビルド

`/app/` で配信されるリファレンス Web クライアントは、Vite のビルド出力です。その
バンドルとビューアエンジンは HOOPS Visualize Web SDK の内容物であり、`@ts3d-hoops`
の npm パッケージとして提供されます。そのため、ビルド出力は
**本リポジトリには含まれません**。ローカルで
ビルドします。ビルドのたびに、それらのパッケージが（各パッケージに同梱される
ライセンステキストとともに）レジストリから取得されます。これは
`bin/scs_points` バイナリがここに含まれず、ローカルでビルドされるのと同じ考え方です。

Node.js が必要です（v22.12.0 LTS で検証）。リポジトリのルートから実行します。

```bash
cd web
npm ci
npm run build
```

これによりクライアントが `static/app/` に書き出されます。このステップがなくても
サーバーは起動し、API も動作します — クライアントをビルドするまで `/app/` だけが
利用できず、上記のコマンドを実行するよう促すメッセージを返します。詳細は
[Web クライアント](#web-クライアント) を参照してください。

### 4. サーバーの起動

**Windows:**
```bat
<HOOPS_AI_INSTALL_DIR>\.venv\Scripts\python.exe main.py --port 8000
```

**Linux (headless server, no display):**
```bash
./start_server.sh --port 8000
```
`start_server.sh` は `:99` に `Xvfb` 仮想ディスプレイを起動し（HOOPS はストリーム
キャッシュをオーサリングするために OpenGL コンテキストを必要とします）、HOOPS AI
venv で `main.py` を実行します。venv のパスを上書きするには `HOOPS_AI_VENV` を設定
します。先に `xvfb` をインストールしてください（`sudo apt-get install -y xvfb`）。

**Linux (desktop session, display present):**
```bash
source /path/to/HOOPS_AI/.venv/bin/activate   # HOOPS AI の venv
python main.py --port 8000
```
実際のデスクトップを持つマシン（例: VirtualBox の GUI セッション）では `$DISPLAY`
が既に設定されているため `Xvfb` は不要です。`start_server.sh` ではなく `main.py` を
直接実行してください。

`main.py` は `--host`（既定 `0.0.0.0`）、`--port`（既定 `8000`）、`--reload` を
受け付けます。ポートが既に使用中の場合、サーバーは起動を拒否します。

---

## Web クライアント

リファレンス Web クライアント（ギャラリー + 3D ビューア）がサーバーに同梱されています。
これはデモであり、ISV が改変できる出発点です。**HOOPS Visualize Web** の上に構築し、
その SDK に付属する Web コンポーネントベースのユーザーインターフェースフレームワークで
ある **HOOPS Web Components** で画面を組んでいます。npm のパッケージで言うと、
`@ts3d-hoops/web-viewer-components` が UI 要素（`<hoops-web-viewer>` 要素とその周りの
ツールバー）を、`@ts3d-hoops/web-viewer` がクライアントから呼ぶビューア API を、
`@ts3d-hoops/sc-engine` がグラフィックスエンジンを提供します。3つとも、サーバー上で
オーサリングされる SCS に合わせて `2026.2.0` に固定されています。

**SDK のファイルは手作業ではなく npm 経由で入ります。** HOOPS Visualize Web の
デプロイでは従来、`hoops-web-viewer.mjs` と `engine.esm.wasm` を SDK のインストール
先から Web ルートへ自分でコピーしていました。本サンプルでは、固定した npm パッケージが
同じファイルを持っています。`npm ci` がそれらを `web/node_modules/` に取得し、
`npm run build` がビューアを `static/app/assets/` にバンドルし、小さな Vite プラグイン
（`web/vite.config.js` の `copyEngineAssets`）が `@ts3d-hoops/sc-engine` から
`engine.esm.wasm` をそのまま `static/app/engine/` にコピーします。CDN からは何も
読み込まず、手作業でコピーするものもありません。

ビルドしたら、`http://<host>:<port>/app/`（例: `http://127.0.0.1:8000/app/`）で
開きます。同一オリジンでサーバーと通信するため、CORS の設定は不要です。

### UI からのインデックス管理

サイドパネルのインデックスピッカーの隣には **New…** と **Delete** ボタンがあり、
オペレーターはブラウザから離れることなく名前付きインデックスを作成・削除できます。

- **New…** は、インデックス名（`^[a-z0-9_-]{1,64}$`）を受け取る小さなダイアログを
  開き、空のインデックスを作成します（`POST /similarity/index/create`）。埋め込み
  モデルはサーバーが起動時に解決した唯一のチェックポイントで、
  ここでは選択しません。名前が既に存在する場合（409）や無効の場合（422）、
  サーバーのメッセージがインラインで表示されます。
- **Delete** は、確認後に選択したインデックスとそのすべてのデータを削除します
  （`DELETE /similarity/index/{name}?confirm=true`）。

作成または削除の後、ピッカーはインデックスを再列挙し、使用可能なものを選択します。
インデックスに部品を追加するには **Register…** を使用します（下記の一括登録
ワークフローを参照）。

### バージョン固定 (3 コンポーネントを同期)

3つの別個の HOOPS コンポーネントは、同じ StreamCache（SCS）データをやり取りするため、
**同じバージョン**（現在は `2026.2.0`）に揃えておく必要があります。

- **HOOPS AI**。サーバー上で部品ごとの SCS をオーサリングします。エクスポートは
  その CAD アクセス層（`viewer.py` から呼び出す
  `hoops_ai.cadaccess.HOOPSTools.exportStreamCache()`）を通して行われます。
  その層が内部で何を使っているかは SDK の中であり、このリポジトリからは確認できません。
- **libsc**。`HOOPS_Visualize_Web_2026.2.0` SDK から `tools/scs_points` に静的
  リンクされ、Shape Map 点群の SCS を書き込みます。これは上の部品ごとの
  エクスポートとは別の経路です。
- ブラウザビューア。`web/package.json` / `web/package-lock.json` で固定された
  `@ts3d-hoops/*` npm パッケージで、クライアント側で SCS を読み込みます。

前の2つが SCS を書き込み、3番目がそれを読み込みます。この関係はコードには
現れないため、バージョンの不一致は目立って失敗せず — モデルを読み込めないビューアと
いう形で後から表面化します。**3つを同時にアップグレードしてください**。npm ビューア
だけを単独で上げてはいけません。

### レイアウト

- `web/` — クライアントのソース（Vite プロジェクト）。読んだり編集したりするのは
  ここです。
- `static/app/` — ビルド出力。**本リポジトリには含まれません**。`npm ci && npm run build` で
  ローカルにビルドします（[セットアップ](#セットアップ) の手順3を参照）。
  再頒布パッケージには、あなたがビルドしたクライアントが含まれます。

### ビルド

クライアントのビルドは **必須のセットアップ手順** です — `static/app/` は本リポジトリ
には含まれないため、クローンしたばかりの状態ではビルドするまでクライアントが
存在しません。`web/` 内を変更したら再ビルドしてください。

**なぜビルド出力がリポジトリに含まれないのか。** `engine.esm.wasm` と、
`static/app/assets/` のバンドルに入るビューアのコードは、**HOOPS Visualize Web SDK の
内容物**です。これらをここにコミットすると、本リポジトリがその SDK を再頒布することに
なります。`libsc` を正規のルートで入手しなければならないのと同じ理由です。SDK の
内容物は、公開リポジトリ経由ではなく、ご自身のライセンス（entitlement）を通じて
入手するものです。ローカルでビルド
することで、すべてのユーザーが npm を経由するため、各ユーザーは SDK パッケージ（と
それに同梱されるライセンステキスト）をレジストリから取得します。
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) を参照してください。コンパイル済みの
HOOPS バイナリの内部に埋め込まれたサードパーティコンポーネント（wasm エンジンと
ブラウザビューア、`bin/scs_points` にリンクされた `libsc`）は npm のメタデータには
現れません。それらの謝辞テキストは [`third_party_notices/`](third_party_notices/)
にあります。

```bash
cd web
npm ci
npm run build
```

これにより、バンドルとビューアエンジン（`engine.esm.wasm`）が `static/app/` に
書き出されます。`static/app/` は `.gitignore` により除外されています。

すべてがローカルで配信されます — クライアントには **外部 CDN への依存がなく**、
一度ビルドすれば完全にオフラインで動作します。

### コンソールハンドル

クライアントは、ブラウザのコンソールから使える 3 つのハンドルを `window` に
公開します。

- `window.Communicator` — `@ts3d-hoops/web-viewer` の名前空間（`Camera`、
  `Point3`、`Color` など）。
- `window.hwv` — 直近に ready になったビューア。
- `window.hwvMap` — マップのビューア。3 種類のビューア（パーツ、比較、マップ）は
  1 つのホスト要素を共有し、それぞれが表示中のビューアを破棄してから自分の
  ビューアを作ります。そのため `window.hwvMap` はマップのビューアが ready に
  なったときに設定され、破棄されたときに null になります — つまりマップが画面に
  出ている間だけ非 null です。マップが無くなると `window.hwv` は黙ってパーツの
  ビューアを指すので、`window.hwvMap` を別に持つ意味はそこにあります。

`Camera` は単なる値なので、マップ表示中に取得したカメラはそのビューアが
置き換えられた後も使えます。設定を変える前後を同じ画角で比較できます。

```js
let cam = hwvMap.view.getCamera();   // マップが画面にある状態で
// Layout を PCA から UMAP に切り替える -- マップは作り直される
hwvMap.view.setCamera(cam);          // 新しいマップのビューアに適用
```

これは開発用の公開であって API ではありません。アプリ側はこれらのハンドルを
一切読んでおらず、予告なく変わることがあります。

### ライセンスキー

追加したビューア機能がブラウザ側のキーを必要とする場合は、`web/src/config.js` に
置いてください。本物のキーはソースツリーの外に置いてください。

---

## Shape Map ポイントクラウド

投影は既定で PCA です。`umap-learn` をインストールすると UMAP も選べるようになります
（[任意: UMAP レイアウト](#任意-umap-レイアウト) を参照）。

ツールバーの **Clustering** セレクタは、部品をどうグループ化するかを選びます。
k-means（既定）か Ward です。どちらもフル次元のベクトル上で動くので、点は 1 つも
動きません。決まるのは、ある部品がどのクラスタに属するかだけです。違いは、Ward が
凝集型の木を 1 本作り、それを要求されたクラスタ数で切る点にあります。そのため、
大きいクラスタ数は、小さいクラスタ数が作った分割を作り直すのではなく **細分** します。
7,928 部品・2,048 次元での実測では、木の構築に約 40 秒かかり、その後はどのクラスタ数
でも数ミリ秒で切れます。したがって最初の構築さえ済めば、クラスタ数を振って見比べる
操作は k-means より Ward のほうが速くなります。追加の依存は不要です。`scipy` は
すでに必須だからです。非常に大きいインデックスは、試みずに拒否します（木は N x N の
距離行列を必要とするため）。上限は `HOOPS_AI_SHAPE_MAP_WARD_MAX_POINTS`（既定 20,000）です。
Ward を選んでいるときの Drill in は階層をたどります。部分集合を新しくクラスタリング
し直すのではなく、同じ木をより深い位置で切るので、ドリルダウンの中のクラスタは、
ドリルしたクラスタの厳密な細分になります。

`POST /similarity/index/{name}/map` は Shape Map を JSON の点として返します。その
マップを HOOPS Visualize Web で 3D 描画するために、サーバーは点群 `.scs` としても配信
できます。

```
GET /similarity/index/{name}/map/scs?color_by=cluster
```

クエリの語彙は `/map`（`n_clusters`、`dims`、`tags`、`tag_mode`、`tagged`）に加え、
焼き込み色のための `color_by`（`cluster` または `tag`）に一致します。`.scs` は `/map`
が返すのと同じキャッシュ済みマップペイロードからオーサリングされるため、ファイル内の
点のインデックスはマップ上の位置と一致します — その順序が、各部品の `file_id` への
リンクになります。

このファイルは小さな C++ ツール `scs_points` によって生成されます。
[`tools/scs_points/`](tools/scs_points/README.md) からビルドします。このツールは
HOOPS Visualize Web のライセンスキーを必要とするため、**本リポジトリには含まれません**。
ビルドして実行ファイルを `bin/`
に配置してください。ツールが存在しない場合、`/map` は動作し続け、`/map/scs` のみが
**503** を返します。生成されたファイルは `scs_map/` 配下に、
`HOOPS_AI_SHAPE_MAP_SCS_CACHE_MAX`（既定 `32`）で設定される LRU 上限付きで
キャッシュされます。

### クライアントでのマップ操作

マップ上部のツールバーは、検索には影響させずに「何を描くか」を変えます。

- **View** — `Point cloud` はマップ対象の部品をすべて点として描画します。
  `3D models` は表示中のカテゴリの点を、上限の範囲で実際の部品モデルに置き換え、
  形状で一族を見分けられるようにします。
- **Scale** / **True scale** — そのモデルの大きさです。通常は各モデルが画面上で
  同じ大きさになるよう正規化されます（埋め込みが見ているものと同じ。L2 正規化は
  絶対寸法を捨て、プロポーションだけを残します）。**True scale** は共通の尺度で
  描画するため、実寸比が見えます。

凡例の行やツリーのノードを右クリックすると **Pin parts** が選べます。そのクラスター
の部品を、他を隠すことなく、すべてモデルとして描画します。ピン留めしたクラスターは
全体を採るか採らないかのどちらかなので、一族が中途半端に表示されることはありません。
**Unpin all** で通常の代表モデル表示に戻ります。

**Catalogue** タブは、Ward のツリーを 1 つの凝集度しきい値で一族に切り分け
（`POST /similarity/index/{name}/map/catalogue`）、一族ごとに 1 つのタイルを表示
します。**Order** はタイルをツリー順または部品数順に並べ替えます。既に読み込み済み
のものをクライアント側で並べ替えるだけなので、リクエストは発生しません。チェック
した一族は 1 回の書き込みでまとめてタグ付けできます。

右側のサイドパネルは、左端のバーをドラッグして幅を変えられるほか、そのバーの矢印で
完全に畳むこともできます。マップにウィンドウ全体を使いたいときや、スクリーンショット
を撮るときに便利です。畳んだ状態でもバーは残り、もう一度クリックすると畳む直前の幅で
復帰します。

---

## API エンドポイント

> **対話的な API リファレンス。** サーバーの実行中は、完全で常に最新のパラメータ一覧が
> 自動生成される OpenAPI ドキュメントに置かれています: **`/docs`**（Swagger UI、
> ブラウザでリクエストを試せます）、**`/redoc`**（リファレンスビュー）、
> **`/openapi.json`**（機械可読スキーマ）— 例: `http://127.0.0.1:8000/docs`。すべての
> クエリパラメータが、その既定値・許容範囲・意味とともにそこに文書化されています。
> 以下の例は最も一般的なオプション（`top_k` など）のみを示しています。
> `/similarity/index/{name}/search` や `.../search-assembly` のようなエンドポイントは
> さらに多く（結果のフィルタリング、タグフィルタ、アセンブリマッチングの調整）を
> 受け付けます。ここでの要約に頼らず、完全な一覧は `/docs` を参照してください。

> **Windows に関する注意:** PowerShell では `curl` は `Invoke-WebRequest` の
> エイリアスであり、以下の `curl` フラグを **理解しません**。すべての例で `curl.exe`
> （Windows 10/11 に同梱される本物のバイナリ）を使用してください。Linux/macOS では
> `curl` を使います。この節の例は Linux 形式の `curl` を使用しています。すぐ下の
> コピー＆ペースト用 **クイックスタート** は両方のシェル向けに用意しています。

### クイックスタート: ターミナルからインデックスを作成して検索

再頒布パッケージには **インデックスが同梱されていません**。そのため、フローは常に
インデックスの作成から始まり、次に CAD ファイルのフォルダをバックグラウンドジョブと
して登録し、そして検索します。フォルダのパスと `my-parts` は必要に応じて置き換えて
ください。

**Windows (PowerShell, `curl.exe`):**

```powershell
$base    = "http://127.0.0.1:8000"
$index   = "my-parts"
$folder  = "C:\cad\parts"          # CAD ファイルのフォルダー (再帰的に検索します)
$workers = 12                       # 並列に動く埋め込みワーカー数
$timeLimit = 120                    # 1 ファイルあたりの埋め込み時間の上限 (秒)

# 1. インデックスを作成します (既にあれば 409 ですが問題ありません)
curl.exe -X POST "$base/similarity/index/create?name=$index"

# 2. CAD ファイルを再帰的に集めます (TMCAD 形式のツリーはサブフォルダーにファイルを持ちます)
#    この一覧はサーバーの CAD_ALLOWED_EXTENSIONS (66 種) と対応しています。
#    有効な CAD ファイルが黙って飛ばされないよう、同期を保ってください。
$exts = @(".3dm",".3ds",".3dxml",".3mf",".arc",".asm",".catdrawing",".catpart",
          ".catproduct",".catshape",".cgr",".dae",".dgn",".dlv",".dwf",".dwfx",
          ".dwg",".dxf",".exp",".fbx",".glb",".gltf",".hsf",".iam",".ifc",
          ".ifczip",".iges",".igs",".ipt",".jt",".kmz",".mf1",".model",".neu",
          ".nwd",".obj",".par",".pkg",".prc",".prt",".psm",".pwd",".rvt",".sab",
          ".sat",".session",".sldasm",".sldfpp",".sldprt",".step",".stl",".stp",
          ".stpz",".u3d",".unv",".vda",".vrml",".wrl",".x_b",".x_t",".xas",
          ".xmt",".xmt_txt",".xpr",".xv0",".xv3")
$cad = Get-ChildItem $folder -Recurse -File |
       Where-Object { $exts -contains $_.Extension.ToLower() }
"CAD files found: $($cad.Count)"

# 3. 各ファイルをアップロードし、返ってくる file_id を集めます
$fileIds = @(); $i = 0
foreach ($f in $cad) {
    $r = curl.exe -s -X POST "$base/files/upload" -F "file=@$($f.FullName)" | ConvertFrom-Json
    if ($r.file_id) { $fileIds += $r.file_id }
    if (++$i % 250 -eq 0) { "uploaded $i / $($cad.Count)" }
}
"collected file_ids: $($fileIds.Count)"

# 4. workers と time_limit を指定して一括登録ジョブを投入します。
#    JSON はファイルに書いて --data-binary "@file" で送ります。-d "{...}" で
#    直接渡すと、PowerShell が内側の二重引用符を取り除いてしまいます。
#    配列を手で組み立てることで、ConvertTo-Json が要素 1 個の配列を
#    スカラーに潰してしまう (API が拒否します) のも避けられます。
$idsJson = ($fileIds | ForEach-Object { $_ | ConvertTo-Json }) -join ','
$payload = "{`"file_ids`":[$idsJson],`"workers`":$workers,`"time_limit`":$timeLimit}"
$tmp = Join-Path $env:TEMP "job_add.json"
Set-Content -Path $tmp -Value $payload -Encoding ascii
$job = curl.exe -s -X POST "$base/similarity/index/$index/jobs/add" `
    -H "Content-Type: application/json" --data-binary "@$tmp" | ConvertFrom-Json
Remove-Item $tmp
$jobId = $job.job_id
"job_id: $jobId"

# 5. ジョブが queued/running でなくなるまでポーリングします
do {
    Start-Sleep 3
    $st = curl.exe -s "$base/similarity/index/$index/jobs/$jobId" | ConvertFrom-Json
    "$($st.status): $($st.progress.done)/$($st.progress.total) (errors $($st.progress.errors))"
} while ($st.status -eq "queued" -or $st.status -eq "running")

# 6. ファイルごとの登録レポートを読みます
curl.exe -s "$base/similarity/index/$index/jobs/$jobId/report"

# 7. いずれかのファイルに似た部品をインデックスから検索します
curl.exe -X POST "$base/similarity/index/$index/search?top_k=10" -F "file=@$($cad[0].FullName)"
```

**Linux / macOS (bash, `curl` + [`jq`](https://jqlang.github.io/jq/)):**

```bash
base=http://127.0.0.1:8000
index=my-parts
folder=/cad/parts                 # CAD ファイルのフォルダー (再帰的に検索します)
workers=12                        # 並列に動く埋め込みワーカー数
time_limit=120                    # 1 ファイルあたりの埋め込み時間の上限 (秒)

# 1. インデックスを作成します (既にあれば 409 ですが問題ありません)
curl -X POST "$base/similarity/index/create?name=$index"

# 2. CAD ファイルを再帰的に集め、順にアップロードして file_id を集めます
#    -iname の一覧はサーバーの CAD_ALLOWED_EXTENSIONS (66 種) と対応しています。
#    有効な CAD ファイルが黙って飛ばされないよう、同期を保ってください。
ids=()
while IFS= read -r -d '' f; do
  id=$(curl -s -X POST "$base/files/upload" -F "file=@$f" | jq -r .file_id)
  [ -n "$id" ] && ids+=("$id")
done < <(find "$folder" -type f \( \
  -iname '*.3dm' -o -iname '*.3ds' -o -iname '*.3dxml' -o -iname '*.3mf' \
  -o -iname '*.arc' -o -iname '*.asm' -o -iname '*.catdrawing' \
  -o -iname '*.catpart' -o -iname '*.catproduct' -o -iname '*.catshape' \
  -o -iname '*.cgr' -o -iname '*.dae' -o -iname '*.dgn' -o -iname '*.dlv' \
  -o -iname '*.dwf' -o -iname '*.dwfx' -o -iname '*.dwg' -o -iname '*.dxf' \
  -o -iname '*.exp' -o -iname '*.fbx' -o -iname '*.glb' -o -iname '*.gltf' \
  -o -iname '*.hsf' -o -iname '*.iam' -o -iname '*.ifc' -o -iname '*.ifczip' \
  -o -iname '*.iges' -o -iname '*.igs' -o -iname '*.ipt' -o -iname '*.jt' \
  -o -iname '*.kmz' -o -iname '*.mf1' -o -iname '*.model' -o -iname '*.neu' \
  -o -iname '*.nwd' -o -iname '*.obj' -o -iname '*.par' -o -iname '*.pkg' \
  -o -iname '*.prc' -o -iname '*.prt' -o -iname '*.psm' -o -iname '*.pwd' \
  -o -iname '*.rvt' -o -iname '*.sab' -o -iname '*.sat' -o -iname '*.session' \
  -o -iname '*.sldasm' -o -iname '*.sldfpp' -o -iname '*.sldprt' \
  -o -iname '*.step' -o -iname '*.stl' -o -iname '*.stp' -o -iname '*.stpz' \
  -o -iname '*.u3d' -o -iname '*.unv' -o -iname '*.vda' -o -iname '*.vrml' \
  -o -iname '*.wrl' -o -iname '*.x_b' -o -iname '*.x_t' -o -iname '*.xas' \
  -o -iname '*.xmt' -o -iname '*.xmt_txt' -o -iname '*.xpr' -o -iname '*.xv0' \
  -o -iname '*.xv3' \
  \) -print0)
echo "collected file_ids: ${#ids[@]}"

# 3. workers と time_limit を指定して一括登録ジョブを投入します
json=$(printf '%s\n' "${ids[@]}" | jq -R . \
  | jq -s --argjson w "$workers" --argjson t "$time_limit" \
       '{file_ids: ., workers: $w, time_limit: $t}')
job_id=$(curl -s -X POST "$base/similarity/index/$index/jobs/add" \
  -H "Content-Type: application/json" -d "$json" | jq -r .job_id)
echo "job_id: $job_id"

# 4. ジョブが queued/running でなくなるまでポーリングします
while :; do
  sleep 3
  s=$(curl -s "$base/similarity/index/$index/jobs/$job_id")
  st=$(echo "$s" | jq -r .status)
  echo "$st: $(echo "$s" | jq -r '.progress.done')/$(echo "$s" | jq -r '.progress.total')"
  [ "$st" = queued ] || [ "$st" = running ] || break
done

# 5. ファイルごとの登録レポートを読みます
curl -s "$base/similarity/index/$index/jobs/$job_id/report"

# 6. クエリファイルに似た部品をインデックスから検索します
curl -X POST "$base/similarity/index/$index/search?top_k=10" -F "file=@/cad/parts/sample.stp"
```

`workers` と `time_limit` は任意です — 省略すると、ワーカー数は自動的に決定され
（`HOOPS_AI_MAX_WORKERS` で上限）、ファイルごとの既定の制限時間が使われます。すべての
オプションについては下記の *一括登録ジョブ* を参照してください。上記の PowerShell の
「JSON をファイル経由で渡す」書き方に注意してください。これは Windows の2つの落とし穴
（PowerShell がインラインの `-d` JSON からクォートを剥がすこと、`ConvertTo-Json` が
要素1つの配列をスカラーに潰すこと）を回避します。

ファイルを1つずつアップロードする代わりにジョブへ入力を与える他の方法: フォルダを
1つの ZIP にまとめてアップロードし、その `zip_file_id` を渡す。あるいは — サーバー
マシン上に既にあるフォルダからオペレーターがシードする場合 — `server_paths` を渡す
（`HOOPS_AI_ALLOW_SERVER_PATHS=true` が必要）。

### ファイル管理

CAD ファイルは一度アップロードされ、以後はファイル内容の SHA-256 16進ダイジェストで
ある `file_id` で参照されます。同じ内容を再アップロードすると、バイト列を再保存する
ことなく同じ `file_id` が返されます。

```
POST   /files/upload                 # 単一ファイル -> { file_id, filename, already_existed }
POST   /files/upload-batch           # 1 回のリクエストで複数ファイル
POST   /files/exists                 # 指定した file_id のうち、既に保存されているもの
POST   /files/upload-from-path       # サーバー側の CAD ファイルまたは ZIP をパスで登録します
GET    /files/                       # 保存済みの CAD ファイルを一覧します
GET    /files/{file_id}/download     # 保存済みの CAD ファイルをダウンロードします (元のファイル名)
POST   /files/{file_id}/scs          # 表示用に、そのファイルの SCS ストリームキャッシュを生成します
DELETE /files/{file_id}              # 保存済みの CAD ファイルを削除します
```

`GET /files/{file_id}/download` は、元のファイル名で生の CAD バイト列を返すため、
クライアントはモデルをローカルで（例えば HOOPS Visualize Web で）レンダリングできます。不明な
id は **404** を返します。

**Windows (PowerShell, `curl.exe`):**

```powershell
# アップロードしてから file_id でダウンロードします
curl.exe -X POST "http://127.0.0.1:8000/files/upload" -F "file=@C:\path\to\model.stp"
curl.exe -o model.stp "http://127.0.0.1:8000/files/<file_id>/download"
```

**Linux / macOS (bash, `curl`):**

```bash
# アップロードしてから file_id でダウンロードします
curl -X POST "http://127.0.0.1:8000/files/upload" -F "file=@/path/to/model.stp"
curl -o model.stp "http://127.0.0.1:8000/files/<file_id>/download"
```

### 形状比較と埋め込み

```
POST /similarity/embed      # 1 ファイルの埋め込みベクトルを計算します
POST /similarity/compare    # 複数ファイルの総当たりの類似度
```

どちらもアップロードおよび／または `file_id` の値を受け付け、アクティブな埋め込み
モデルを使用します。

### 名前付きインデックス管理

独自の永続インデックスを構築し、時間をかけて成長させられます。

```
POST   /similarity/index/create               # 空のインデックスを作成します
GET    /similarity/index/list                 # インデックスを一覧します
POST   /similarity/index/add                   # 部品を同期的に登録します
POST   /similarity/index/{name}/search         # インデックスから似た部品を検索します
POST   /similarity/index/{name}/warm           # クエリに対する検索応答キャッシュを事前に温めます
POST   /similarity/index/{name}/search-assembly# アセンブリ同士の検索
GET    /similarity/index/{name}/stats          # ファイル / ボディ / アセンブリの件数
POST   /similarity/index/{name}/map            # インデックスの 2D/3D 類似マップ
GET    /similarity/index/{name}/parts           # 登録済みの部品を一覧します (アセットの URL 付き)
DELETE /similarity/index/{name}/parts           # 部品を取り除きます
DELETE /similarity/index/{name}?confirm=true    # インデックス全体を削除します
```

**Windows (PowerShell, `curl.exe`):**

```powershell
# 名前付きインデックス (部品) を、クエリ自身も含めて検索します
curl.exe -X POST "http://127.0.0.1:8000/similarity/index/my-parts/search?top_k=10&file_id=<file_id>"

# 「このサブアセンブリを含むアセンブリはどれか」
curl.exe -X POST "http://127.0.0.1:8000/similarity/index/my-parts/search-assembly?top_k=10" `
  -F "file=@C:\path\to\assembly.stp"
```

**Linux / macOS (bash, `curl`):**

```bash
# 名前付きインデックス (部品) を、クエリ自身も含めて検索します
curl -X POST "http://127.0.0.1:8000/similarity/index/my-parts/search?top_k=10&file_id=<file_id>"

# 「このサブアセンブリを含むアセンブリはどれか」
curl -X POST "http://127.0.0.1:8000/similarity/index/my-parts/search-assembly?top_k=10" \
  -F "file=@/path/to/assembly.stp"
```

部品を削除すると、その `thumbnails/<id>.png` と `scs/<id>.scs` のアセットも削除され、
タグも整理されるため、孤立するものは残りません。

`POST /similarity/index/{name}/warm` は、あるクエリに対して `search` とまったく同じ
経路を実行し、ヒットは破棄して応答キャッシュだけを満たします。これにより最初の実際の
リクエストがウォームな状態で処理されます。パラメータは後続の検索と **完全に** 一致
していなければなりません（`top_k`、`kind`、`include_self`、タグフィルタはすべて
キャッシュキーの一部です）。さもないとウォームは別のキーを書き込み、実際の検索は
決してヒットしません。キャッシュが有効（`HOOPS_AI_SEARCH_CACHE`）である必要があり、
応答は `cache_enabled` を報告するため、無効化されたウォームが判別できます。

### 部品別アセット (サムネイル / ストリームキャッシュ)

```
GET /similarity/index/{name}/parts/{part_id}/thumbnail   -> image/png
GET /similarity/index/{name}/parts/{part_id}/scs         -> application/octet-stream
```

登録時に生成される、部品ごとの PNG サムネイル（ギャラリー表示用）または SCS ストリーム
キャッシュ（Web クライアントがビューアに読み込むもの）を配信します。アセットが
存在しない場合は **404** を返します。パスは `indexes/<name>/` 内に収まるよう検証され
ます。

**Windows (PowerShell, `curl.exe`):**

```powershell
curl.exe -o thumb.png "http://127.0.0.1:8000/similarity/index/my-parts/parts/<file_id>/thumbnail"
curl.exe -o part.scs  "http://127.0.0.1:8000/similarity/index/my-parts/parts/<file_id>/scs"
```

**Linux / macOS (bash, `curl`):**

```bash
curl -o thumb.png "http://127.0.0.1:8000/similarity/index/my-parts/parts/<file_id>/thumbnail"
curl -o part.scs  "http://127.0.0.1:8000/similarity/index/my-parts/parts/<file_id>/scs"
```

### Shape Map

```
POST /similarity/index/{name}/map   -> 2D/3D similarity map of the index
```

インデックスの部品を類似度マップに射影します。保存された body ごとの埋め込みを
インデックスから読み戻し（CAD ファイルの再埋め込みは行いません）、ファイルごとに1点へ
集約し（そのファイルの body の平均を再正規化）、PCA で射影し、FAISS の k-means で
色付けします。類似する部品どうしは近くに配置されます。座標は `[-1, 1]` の立方体へ
正規化され、与えられたインデックスとパラメータに対してレイアウトは **決定的** です
（k-means のシードは固定）。そのため、安定したレンダリング入力として安全に使えます。

クエリパラメータ:

| パラメータ | 既定 | 意味 |
|-------|---------|---------|
| `n_clusters` | `0` | k-means のクラスタ数。`0` は `min(max(2, N // 50), 24)` を自動選択します（点数ヒューリスティック）。`-1` は `[2, min(24, N-1)]` の範囲で `k` を走査し、イナーシャ曲線のエルボーを選びます（データ駆動、重め）。正の値はその `k` を厳密に強制します。件数が固定なら変更コストは小さいです（キャッシュされた射影を再利用）。Web UI は既定 `8` で、**Optimal** オプション（`-1`）を提供します。 |
| `dims` | `3` | 射影の次元数、`2` または `3`。未使用の軸は `0` です。変更コストは小さいです。 |
| `tags` | — | カンマ区切りのタグ。マップを一致する部品に限定し、そのサブセットのみでレイアウトを再計算します。 |
| `tag_mode` | `any` | `any` は少なくとも1つのタグに一致、`all` はすべてのタグを要求します。 |
| `tagged` | — | `true` = タグ付きの部品のみ、`false` = タグなしのみ。 |
| `include_tags` | `false` | 各部品のタグをすべての点に追加します。応答を小さく保つため既定はオフです。 |

応答は `points`（ファイルごとに1つの `{file_id, filename, x, y, z, cluster}`）と、
マップされたサブセットをタグ別（サブセットにタグがない場合はクラスタ別）に要約した
サーバー側の `legend` を含みます。あるインデックスに対する最初の呼び出しは射影を実行し
（大きなコーパスで数秒）、それをキャッシュします。以降の呼び出しは、異なる
`n_clusters` / `dims` を含めて、キャッシュから素早く返ります。登録ジョブの後に
ウォームするには、上記の `HOOPS_AI_SHAPE_MAP_PREWARM` を参照してください。

**Windows (PowerShell, `curl.exe`):**

```powershell
# 既定の 3D マップ、クラスター数は自動
curl.exe -X POST "http://127.0.0.1:8000/similarity/index/my-parts/map"

# タグ付き部品だけの 2D マップ、12 クラスター、点ごとのタグ付き
curl.exe -X POST "http://127.0.0.1:8000/similarity/index/my-parts/map?dims=2&n_clusters=12&tagged=true&include_tags=true"
```

**Linux / macOS (bash, `curl`):**

```bash
# 既定の 3D マップ、クラスター数は自動
curl -X POST "http://127.0.0.1:8000/similarity/index/my-parts/map"

# タグ付き部品だけの 2D マップ、12 クラスター、点ごとのタグ付き
curl -X POST "http://127.0.0.1:8000/similarity/index/my-parts/map?dims=2&n_clusters=12&tagged=true&include_tags=true"
```

```
POST /similarity/index/{name}/map/catalogue  -> Ward ツリーから切り出した部品の一族
```

インデックスの Ward ツリーを 1 つの凝集度で一族に切り分けます。各枝は、凝集度が
`threshold`（既定 `0.95`）に達した最初のノードで切られ、最後まで達しない枝は捨てられる
のではなく `reached=false` として返されます。そのため部品の合計は常に母集団全体を
覆います。`k_max`（既定 `2048`）はフロンティアを切り出すツリーの深さ、`min_parts` は
それより小さいグループを `groups` から除外します（除外されたものは
`omitted_by_min_parts` に残ります）。`min_reached_parts`（既定 `5`）は、しきい値に
達したグループを *small* ではなく *confident* と見なす大きさで、分類のための値に
過ぎず、フロンティアを構成するノードを変えることはありません。`kind` / `tags` /
`tag_mode` / `tagged` は `/map` とまったく同じように母集団を選択します。

### タグ

登録済みの部品に自由形式のタグを付与し、問い合わせます。書き込みは ETag による
楽観的ロックで保護されます。

```
GET    /similarity/index/{name}/tags                       # インデックス内のすべてのタグ
GET    /similarity/index/{name}/parts/{part_id}/tags       # 1 部品のタグ
PUT    /similarity/index/{name}/parts/{part_id}/tags       # 1 部品のタグを置き換えます
POST   /similarity/index/{name}/tags/{tag}/parts           # 複数の部品に一括でタグを付けます
DELETE /similarity/index/{name}/tags/{tag}/parts           # 複数の部品からタグを一括で外します
DELETE /similarity/index/{name}/tags/{tag}?confirm=true    # そのタグをどこからでも廃止します
```

サーバーはアプリケーション全体で1つの **タグ語彙**（データルート直下の
`tag_vocabulary.json`）を保持します。これは「使う名前」の一覧で、どの部品に付いて
いるかとは別に管理されます。名前はどのインデックスであれ最初に割り当てられた時点で
追加され、最後の部品からタグを外しても残るため、名前を記憶から打ち直す必要が
ありません。ファイルは `{"tags": [...]}` の単純な配列で、手で編集できます。
インデックスごとに分けないのは、タグの名前はインデックスの属性ではなく語彙の選択
だからです。候補を出すときは「使用中のタグ」を先に、語彙だけの名前を後に並べて
ください。

```
GET    /similarity/tag-vocabulary                          # 選択候補となる名前の一覧
POST   /similarity/tag-vocabulary                          # 事前に名前を登録します
DELETE /similarity/tag-vocabulary/{tag}                    # 名前を1件削除（部品のタグは残ります）
```

検索エンドポイントはタグフィルタを受け付けるため、結果をタグ付きの部品に絞り込め
ます。

API は `tag_mode=all`（複数タグにまたがる AND）をサポートします。このリファレンス UI
は簡潔さのため単一選択フィルタを使っており、そのため常に `any` を送信します。AND
フィルタリングが必要な場合は、独自の UI を `tag_mode` に接続してください。

### コンテキスト (生産レコード)

登録済みの部品に生産レコード（材料、工程、コスト、リードタイムなど）を付与し、新しい
部品の欠けているレコードを、その最近傍の部品のレコードから予測します。タグと同様に、
レコードはインデックスごとのサイドカー（`indexes/<name>/context.json`）に置かれ、
`.faiss` ファイルには入りません。これらをベクトルストアに書き込むと `save()` が
呼ばれてインデックスの mtime が動き、次の検索時に約 19.3 秒のアセンブリマッチャ再構築を
強制してしまうためです。読み取りは `ETag` を持ち、書き込みは任意の `If-Match` を
受け付けます。

以下の API は常に利用できます。これを操作する Web クライアントの **Estimate**
パネルはデモ専用で、`HOOPS_AI_DEMO` を設定したときだけ表示されます。予測が読む
に値するのは、インデックスに実際の生産レコードが付いているときであり、デモは
捏造したレコードで動いているためです。

```
POST   /similarity/index/{name}/predict-context               # クエリ部品のフィールドを予測します
GET    /similarity/index/{name}/context                       # どのフィールドがあり、それぞれ何部品が使っているか
GET    /similarity/index/{name}/parts/{part_id}/context       # 1 部品のレコード
PUT    /similarity/index/{name}/parts/{part_id}/context       # 1 部品のレコードを置き換えます
```

`predict-context` は JSON ボディを受け取ります: `file_id`（以前にアップロード／登録
済みの部品）、予測する `keys`、および任意の `top_k`（ポーリングする最近傍数）、タグ
フィルタ、既知のフィールドからなる `query_context`、`neighbor_ids` です。クエリ部品は
`/search` と同じ方法で検索され、自己ヒットは除外され、レコードを持つ `top_k` 個の
最近傍が集約され、キーごとに1つの値が返されます。`neighbor_ids` が与えられた場合、
近傍はちょうどそれらの id に固定されます（Web クライアントは Results タブに表示中の、
類似度しきい値とタグフィルタで既に絞り込まれた部品を送信します）。その場合、サーバーの
プールにそれらが含まれ続けるよう `top_k` はそれらを生成した検索と一致させるべきで、
タグフィルタと `top_k` の上限は再適用されません。応答は `candidates_examined`、
`hits_considered`、`hits_with_context`、`coverage`、`predictions`、および根拠となる
`hits` を報告します。

**Windows (PowerShell, `curl.exe`):**

```powershell
curl.exe -X POST "http://127.0.0.1:8000/similarity/index/my-parts/predict-context" `
  -H "Content-Type: application/json" `
  -d '{"file_id":"<id>","keys":["Material","UnitCost","LeadTimeDays"],"top_k":50}'
```

**Linux / macOS (bash, `curl`):**

```bash
curl -X POST "http://127.0.0.1:8000/similarity/index/my-parts/predict-context" \
  -H "Content-Type: application/json" \
  -d '{"file_id":"<id>","keys":["Material","UnitCost","LeadTimeDays"],"top_k":50}'
```

集約は、任意の `indexes/<name>/context_rules.json`（カテゴリ／数値のルール、キー
ごとのオーバーライド、名前付き正規化器のみ — コードは決して含めない）でインデックス
ごとに調整できます。未知のオプションは、予測を失敗させるのではなく無視されます。

### 一括登録ジョブ

大規模なコーパスを非同期で登録します。クライアントはフォルダをローカルでハッシュし
（`POST /files/exists`）、欠けているものだけをアップロードしてからジョブを投入でき
ます。

```
POST   /similarity/index/{name}/jobs/add            # 一括登録ジョブを投入します
GET    /similarity/index/{name}/jobs                # インデックスのジョブを一覧します
GET    /similarity/index/{name}/jobs/{job_id}       # ジョブをポーリングします
GET    /similarity/index/{name}/jobs/{job_id}/report# ファイルごとの登録レポート
DELETE /similarity/index/{name}/jobs/{job_id}       # ジョブをキャンセルします
```

`POST .../jobs/add` は JSON ボディを受け取ります。入力ソースを **1つ** に加えて、
任意のチューニングを指定します。

| フィールド | 型 | 意味 |
|---|---|---|
| `file_ids` | `string[]` | 既にストアにアップロード済みの CAD ファイルの `file_id`（標準的な経路）。 |
| `zip_file_id` | `string` | 展開して登録する、アップロード済み ZIP アーカイブの `file_id`。 |
| `server_paths` | `string[]` | **サーバーマシン上** のパス。オペレーターのシードのみ。`HOOPS_AI_ALLOW_SERVER_PATHS=true` でない限り **403** を返します。 |
| `workers` | `int` | 並列埋め込みワーカー数。省略すると CPU コア数と空き RAM から自動でサイズ決定します（`HOOPS_AI_MAX_WORKERS`、既定 8 で上限）。 |
| `time_limit` | `int` | ファイルごとの埋め込み予算（秒）。省略すると `HOOPS_AI_EMBED_TIME_LIMIT`、それも無ければ SDK の 120 秒を使います。超過したファイルは `retryable=true` として報告されます。 |

ジョブはすべての入力に対して1回の埋め込みパスを実行し、`job_id` を付けて即座に
返ります。`progress` については `GET .../jobs/{job_id}` をポーリングし、完了時には
`report_url` を取得します。ジョブは既定では1つずつ実行されるため
（`HOOPS_AI_JOB_MAX_CONCURRENCY`）、ジョブはしばらく `queued` に留まることがあります。
ステータスは `queued` → `running` → `done` / `failed` / `canceled` です。

**Windows (PowerShell, `curl.exe`):**

```powershell
# アップロード済みのファイルを登録します。ワーカー 6、1 ファイル 300 秒
curl.exe -X POST "http://127.0.0.1:8000/similarity/index/my-parts/jobs/add" `
  -H "Content-Type: application/json" `
  -d '{"file_ids":["<id1>","<id2>"],"workers":6,"time_limit":300}'

# アップロードしたフォルダーの ZIP を登録します
curl.exe -X POST "http://127.0.0.1:8000/similarity/index/my-parts/jobs/add" `
  -H "Content-Type: application/json" `
  -d '{"zip_file_id":"<zip_file_id>"}'
```

JSON はシングルクォートで囲み、PowerShell が `curl.exe` にそのまま渡すようにします。
変数から組み立てたペイロード（例えば長い `file_ids` 配列）の場合は、代わりにファイル
へ書き出して `--data-binary "@file"` で送信してください — 上記のクイックスタートを
参照してください。

**Linux / macOS (bash, `curl`):**

```bash
# アップロード済みのファイルを登録します。ワーカー 6、1 ファイル 300 秒
curl -X POST "http://127.0.0.1:8000/similarity/index/my-parts/jobs/add" \
  -H "Content-Type: application/json" \
  -d '{"file_ids":["<id1>","<id2>"],"workers":6,"time_limit":300}'

# アップロードしたフォルダーの ZIP を登録します
curl -X POST "http://127.0.0.1:8000/similarity/index/my-parts/jobs/add" \
  -H "Content-Type: application/json" \
  -d '{"zip_file_id":"<zip_file_id>"}'
```

失敗したファイルには `retryable` フラグが付与されます。大規模なコーパスでは、
2パス戦略を実行してください。まず多数の `workers` と短い `time_limit` ですべてを
投入し、次に `retryable` が true の失敗分を、少数の `workers` と長い `time_limit` の
2つ目のジョブとして再投入します。再試行不可の失敗は決定的な CAD エラーであり、
予算をどれだけ与えても決して成功しません。キャンセルはジョブが埋め込みを開始する
**前** にのみ有効になるため、大規模なコーパスは複数のジョブに分割して、キャンセルと
コミットの粒度を制御下に置いてください。

---

## データディレクトリ

| ディレクトリ | 内容 | サーバーが消去する？ |
|---|---|---|
| `uploads/` | CAD ストア — 登録された全ファイルの唯一のコピー。`file_id` をキーとします。 | しない。 |
| `indexes/` | 名前付き FAISS インデックスと、部品ごとの `thumbnails/` / `scs/` アセット、`tags.json`。 | しない（部品ごとのアセットはその部品とともに削除されます）。 |
| `embeddings_cache/` | キャッシュされた埋め込み。 | TTL で掃引（`HOOPS_AI_EMBEDDINGS_CACHE_TTL_DAYS`、既定は永久保持）。 |
| `out/` | 一時的な結果画像と形状マップ。 | TTL で掃引（`HOOPS_AI_OUT_TTL_HOURS`、既定 24 時間）。 |
| `jobs/` | 登録ジョブのレコード。 | `HOOPS_AI_JOB_TTL_DAYS` / `HOOPS_AI_JOB_MAX_RECORDS` で制限。 |
| `models/` | 埋め込みチェックポイント。運用者が自分の `*.ckpt` を置く場所です。`<APP_ROOT>/models/` に同梱された既定より先に検索され、それを隠します（[カスタム (fine-tuned) モデル](#カスタム-fine-tuned-モデル) を参照）。 | しない（サーバーは読み取るだけです）。 |
| `scs_map/` | Shape Map 点群 `.scs` のキャッシュ。 | `HOOPS_AI_SHAPE_MAP_SCS_CACHE_MAX`（既定 32）による LRU で制限。 |
| `tag_vocabulary.json` | 選択候補となるタグ名。全インデックスで共有されます（ディレクトリではなくファイル）。 | しない（名前は明示的な削除時のみ消えます）。 |

これらはすべて単一の **データルート** の下に置かれます。`HOOPS_AI_DATA_DIR` が未設定の
場合、データルートは既定で `<app>/data` — アプリケーションの `data/` サブディレクトリ
— になります。そのため実行時ディレクトリ（`uploads/`、`indexes/` など）はソースツリー
から分離され、新規チェックアウトが設定なしで動作します。データを別のボリュームに置く
には `HOOPS_AI_DATA_DIR` を設定してください（下記の *デプロイ* を参照）。

---

## デプロイ

このサーバーは、HOOPS AI がインストールされて **いない** エンドユーザーへ再配布する
ためにパッケージ化できます。ビルドスクリプトは、SDK、コンパイル済みライセンス shim、
既定の埋め込みモデル、およびサーバーが実際に import する Python パッケージだけを
同梱します。

### パッケージレイアウト

ビルドスクリプトは出力ディレクトリに2つのものを書き出します。組み立てたステージング
ディレクトリと、そのディレクトリの zip です。配布するのは zip で、ステージング
ディレクトリは中身を確認できるよう残されます。

```
<out-dir>/
├── redist_package_<yyyyMMdd-HHmmss>.zip     # 配布する成果物 (展開すると redist_package/ になります)
└── redist_package/                          # ステージングディレクトリ (確認用に残されます)
    ├── *.py                                 # リポジトリルートのサーバーモジュールすべて
    │                                        #   (main.py, core.py, search.py, assembly.py, …)
    ├── requirements.txt, start_server.sh
    ├── routers/  vendor/  static/           # static/ にビルド済み Web クライアント (static/app/) が入ります
    ├── bin/
    │   └── scs_points.exe                   # Linux では scs_points。Shape Map の点群ツール
    ├── hoops_license_shim.<abi-tag>.pyd     # Linux では .so
    ├── .env                                 # テンプレート。設定はすべて任意で、ライセンスキーは含みません
    ├── models/
    │   └── <default>.ckpt                   # 同梱される既定の埋め込みチェックポイント
    ├── .venv/
    │   └── Lib/site-packages/ (win) | lib/python3.12/site-packages/ (linux)
    │                                        # トレースされた HOOPS AI と依存パッケージのみ
    │                                        # (site-packages のみ — Python インタープリタは含みません)
    ├── samples/                             # 任意のサンプル CAD ファイル
    ├── third_party_notices/                 # HOOPS バイナリ内部のコンポーネントに関する通知
    ├── DISCLAIMER.txt                       # サンプル / 概念実証であることの注意書き
    ├── LICENSE                              # ここで書かれた code に対する MIT ライセンス
    ├── THIRD-PARTY-NOTICES.md               # Web クライアントの npm 依存関係のライセンス通知
    └── THIRD_PARTY_MANIFEST.txt             # 同梱パッケージの出所 / 再頒布カテゴリ
```

サーバーモジュールは列挙ではなく **収集** されます。スクリプトはリポジトリルートの
`*.py` を、明示的な除外リスト（テスト用・使い捨てのファイルと、キーを平文で持つ
ライセンス shim のソース）を除いてすべてコピーします。後のリファクタリングで追加された
モジュールはスクリプトを変更しなくても同梱され、サーバーが import するものが欠ければ
`tests/test_redist_packaging.py` が失敗します。

`models/` ディレクトリは2箇所あり、役割が異なります。上のパッケージ内の
`<APP_ROOT>/models/` は **ビルドスクリプトが同梱する既定のチェックポイント** を置く場所
です。`<DATA_ROOT>/models/`（既定では `<app>/data/models/`）は **運用者が自分の
チェックポイントを置く** 場所で、こちらが先に検索され、同梱のものを隠します（隠した場合は
サーバーがログに記録します）。後述の *カスタム (fine-tuned) モデル* を
参照してください。

`main.py` は同梱の `.venv` の site-packages を `sys.path` の先頭に追加するため、
サーバーはシステムへのインストールなしに同梱の HOOPS AI を import します。開発用
チェックアウトではそのディレクトリは存在せず、代わりに自分の環境が使われます。

パッケージは Python インタプリタを **同梱しません** — site-packages のみです。
ターゲットマシンには **Python 3.12** ランタイムが必要です（同梱パッケージとライセンス
shim がビルドされたのと同じマイナーバージョン。ネイティブ拡張と ABI タグ付きの
`.pyd`/`.so` は別のマイナーバージョンでは読み込めません）。

同梱の `.venv` は **トレースされた HOOPS AI SDK とそのネイティブ依存**（`hoops_ai`、
`faiss`、`torch` など）のみを含みます。`requirements.txt` に列挙された軽量な Web
サーバー依存（`fastapi`、`uvicorn`、`python-multipart`、`scipy`）は同梱 **されず**、
サーバーを実行する Python 3.12 ランタイムに一度インストールする必要があります。
その後 `main.py` は Web スタックをランタイムから、SDK を同梱の `.venv` から import
します。

```powershell
# Windows: マシン上の任意の Python 3.12 を使います (例: py ランチャー)
cd redist_package
py -3.12 -m pip install -r requirements.txt   # 初回のみ: Web サーバーの依存関係を導入します
py -3.12 main.py --port 8000
```

```bash
# Linux: Web サーバーの依存関係用に Python 3.12 の venv を作り、そこから実行します。
# Ubuntu 24.04 のシステム Python は外部管理 (PEP 668) で、pip や venv を
# 含まないことが多いため、venv が素直な方法です。venv の作成が
# "No module named venv/pip" で失敗する場合: sudo apt install -y python3-pip python3.12-venv
cd redist_package
python3.12 -m venv .webvenv
source .webvenv/bin/activate
pip install -r requirements.txt              # 初回のみ: Web サーバーの依存関係を導入します
python main.py --port 8000
#
# ヘッドレスサーバー (ディスプレイなし): HOOPS は OpenGL コンテキストを必要とするため、
# main.py を直接実行すると segfault します ("HOOPS Converter failed with exit code -11")。
# 代わりに start_server.sh で仮想ディスプレイの下から起動してください
# (依存関係の導入は一度だけ):
#   sudo apt-get install -y xvfb libgl1 libglu1-mesa libxrender1 libxext6 libsm6
#   HOOPS_AI_VENV="$PWD/.webvenv" ./start_server.sh --port 8000
```

同梱の SDK `.venv` はアクティベートする必要が一切ありません。`main.py` が自身で
`.venv/.../site-packages` を `sys.path` に追加するため、`import hoops_ai` は同梱側に、
`import fastapi` はランタイム側（Windows ではシステムの Python 3.12、Linux では上記の
`.webvenv`）に解決されます。

### 設定 (`.env`)

設定はすべて **任意** です。`.env` はインストーラが既定値を上書きする場所です。
`.env` が全く無くても、サーバーは導出された既定値で起動します。優先順位は
`os.environ` > `.env` > 導出された既定値 — ただし `HOOPS_AI_LICENSE` は例外で、
`.env` からのみ読み込まれ、プロセス環境からは決して読み込まれません。

| 変数 | 未設定時の導出値… |
|---|---|
| `HOOPS_AI_DATA_DIR` | `<app>/data`（`uploads/ indexes/ …` の実行時ディレクトリは `data/` の下に置かれ、ソースと分離されます）。 |
| `HOOPS_AI_SDK_DIR` | 未設定。同梱チェックポイントは使われず、データルート下の名前付きインデックスは引き続き動作します。 |
| `HOOPS_AI_EMBEDDINGS_MODEL_NAME` | `<DATA_ROOT>/models/` 内、無ければ `<APP_ROOT>/models/` 内で見つかった単一の `*.ckpt`。 |
| `HOOPS_AI_LICENSE` | コンパイル済み shim から供給。開発専用の `.env` フォールバックがあります。 |

起動時にサーバーは INFO レベルで、どの値がどこから来たか、解決されたデータルート、
解決された埋め込みチェックポイントの絶対パスとサイズをログに出します — これらの行を
デプロイの診断に使ってください。

### 別ボリューム上のデータ

ターゲットボリューム上の書き込み可能なディレクトリを `HOOPS_AI_DATA_DIR` に設定
します。サーバーは起動時にデータルートが書き込み可能かを確認し、そうでない場合は
（例えばアプリが `Program Files` 配下にインストールされ、データディレクトリが未設定
の場合）、最初のアップロードで失敗するのではなく、明確なエラーで停止します。既存の
データが移動・削除されることは決してありません。既存のストアを再利用するには、
`HOOPS_AI_DATA_DIR` をそこに向けてください。

### カスタム (fine-tuned) モデル

単一の `*.ckpt` を `<DATA_ROOT>/models/` に置くと、`<APP_ROOT>/models/` の同梱既定
モデルを上書きできます。データルートのモデルが同梱モデルを隠す場合、サーバーはその
事実をログに出します。1つのディレクトリにつき `*.ckpt` はちょうど1つに保ってください。
同じディレクトリに2つ以上あると、候補を列挙した決定的な起動エラーになります。

> モデル／インデックスの整合性: 各インデックスは、それがビルドされたチェックポイントの
> 同一性（SHA-256、サイズ、パス、埋め込み次元）を `index.json` に記録し、すべての
> レコードがチェックポイントのハッシュを保持します。このリリースはその同一性を *記録*
> するだけで、不一致の警告はまだ行いません。モデルを入れ替える前に、ログに出た
> チェックポイントのパス／サイズが、インデックスがビルドされたモデルと一致することを
> 確認してください。

### パッケージのビルド (運用者の手順)

コンパイル済みライセンス shim が先に存在し、Web クライアントがビルド済みであり
（`cd web; npm ci; npm run build`）、`bin/scs_points[.exe]` が存在している必要があります
（*ライセンスキーの扱い* を参照）。いずれかが欠けているとスクリプトは停止します。
Linux では `zip` がインストールされている必要があります。スクリプトはコピーを始める前に
これを確認するため、数ギガバイトのビルドが最後の工程で無駄になることはありません。
次に、使用されるモジュールをトレースし、パッケージをビルドします。

```powershell
# Windows PowerShell
$env:HOOPS_AI_HOME = "C:\SDK\HOOPS_AI\V1.1"
$env:HOOPS_AI_LICENSE = '<your-key>'   # シングルクォート: PowerShell は $ を展開しません
python tools/trace_modules.py `
    --cad-file C:\path\to\sample.step `
    --embed-ckpt "$env:HOOPS_AI_HOME\packages\trained_ml_models\ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt" `
    --out C:\temp\used_top_level_modules.txt
.\tools\build_redist_package.ps1 `
    -ModulesFile C:\temp\used_top_level_modules.txt `
    -Ckpt "$env:HOOPS_AI_HOME\packages\trained_ml_models\ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt" `
    -OutDir C:\temp\redist_out
```

```bash
# Linux (bash)
export HOOPS_AI_HOME="$HOME/SDK/HOOPS_AI/V1.1"
export HOOPS_AI_LICENSE='<your-key>'   # シングルクォート: シェルは $ を展開しません
python3 tools/trace_modules.py \
    --cad-file /path/to/sample.step \
    --embed-ckpt "$HOOPS_AI_HOME/packages/trained_ml_models/ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt" \
    --out /tmp/used_top_level_modules.txt
tools/build_redist_package.sh \
    --modules-file /tmp/used_top_level_modules.txt \
    --ckpt "$HOOPS_AI_HOME/packages/trained_ml_models/ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt" \
    --out-dir /tmp/redist_out
```

ビルドスクリプトはコンパイル済み shim を必須とし、`HOOPS_AI_LICENSE` 行を含む `.env`
のステージングを拒否するため、平文キーがパッケージに紛れ込むことは決してありません。
実行が終わると `<out-dir>/redist_package_<yyyyMMdd-HHmmss>.zip` ができます。これが配布
する成果物です。タイムスタンプが付くため、2回のビルドが黙って互いを上書きすることは
ありません。また `redist_package/` ステージングディレクトリは zip の隣に残されるため、
何が入ったかを確認できます。（Windows PowerShell 5.1 の `Compress-Archive` は 2 GB を
超えるメンバーファイルを扱えません。この理由で圧縮に失敗しても、ステージング
ディレクトリは完成しています。PowerShell 7+ で再実行するか、独自のアーカイバで圧縮して
ください。）`THIRD_PARTY_MANIFEST.txt` には同梱パッケージ
が列挙されます。Tech Soft 3D パッケージの再配布は、あなたの TPA に基づいて個別に確認
する必要があります。

`tools/trace_modules.py` の `run_traced_workload()` は、サーバーが呼び出しうるすべての
機能を実行しなければなりません — `hoops_ai.ml.context_layer` を import する
`predict-context` を含みます — そうすることでパッケージがそれらの機能に必要な
モジュールを同梱します。`run_traced_workload()` を変更した場合は、再トレースして
パッケージを再ビルドしてください。

### Python バージョンとプラットフォーム

同梱の `.venv` とライセンス shim は、1つの Python マイナーバージョンとプラットフォーム
向けにビルドされます。Python ランタイムをアップグレードすると（例: 3.12 → 3.13）、
ABI タグ付きのライセンス shim は import 不能になり — サーバーは警告付きで `.env`
フォールバックに戻ります — したがって **Python をアップグレードした後は shim を再ビルド
し、パッケージを再トレース／再ビルドしてください**。Linux と Windows はそれぞれ独自の
ビルドが必要です。

---

## 運用

### メモリサイジング

サーバーはいくつかのものをプロセスメモリに常駐させます。ホストのサイズはまず読み取り
側から見積もり、次に書き込みキャッシュを加えてください。

| コンポーネント | おおよその常駐サイズ（`tmcad` クラスのインデックス: 10,230 ファイル / 39,377 body / 次元 2048） |
|-----------|----------------------------------------------------------------------------------------|
| 名前付きサーチャー | 各 ~308 MB（`index.faiss`）、最大 2 個保持 |
| アセンブリマッチャ | ~616 MB（サーチャー + 全 body ベクトルの L2 正規化コピー） |
| Shape Map キャッシュ | ~84 MB（集約行列1つ。結果は数 MB） |
| 埋め込みモデル | 数百 MB（単一のチェックポイントを一度ロード） |
| **読み取り側の小計** | **~1.9 GB** |
| 名前付きインデックスの **書き込み** ストア | 最大 `HOOPS_AI_INDEX_CACHE_MAX_MB`（既定 **1024 MB**、≈ `tmcad` インデックス3つ分） |

つまり単一の `tmcad` クラスのインデックスは定常状態でおよそ **3 GB** を必要とし、その
うち約 **1 GB は書き込みキャッシュ** で、調整可能です。メモリが制約されたホストでは
`HOOPS_AI_INDEX_CACHE_MAX_MB` を下げてください（退避されたインデックスは次の
create/add/remove 時にディスクから再ロードされ、数秒 — 数分かかる登録ジョブの中では
無視できます）。複数のインデックスを同時に更新する場合は上げてください。

プロセス内の両キャッシュは、エントリ数ではなく **メモリ** で制限されます。インデックス
サイズはデプロイごとに大きく異なるためです。

- `HOOPS_AI_INDEX_CACHE_MAX_MB`（既定 1024）— 名前付きインデックスの書き込みストア。
- `HOOPS_AI_EMBED_CACHE_MAX_MB`（既定 256）— 埋め込みキャッシュ（ディスク裏付けあり
  のため、退避は再計算ではなく再ロード）。

すべての退避は `[CACHE]` の下で `INFO` レベルでログに出るため（どのインデックス／
エントリが落とされたか、新しい常駐合計）、メモリやレイテンシの問題はログだけから診断
できます。

### 起動時ウォームアップ

インデックスに触れる最初のリクエストは、一度きりのビルドコストを支払います。名前付き
サーチャー（~0.3 秒）、Shape Map の SVD（~15 秒）、アセンブリマッチャの k-means
（~20 秒）です。再起動後は最初のユーザーが再びこれを支払います。起動時ウォームアップは
これらのビルドをバックグラウンドで行い、そのユーザーが支払わずに済むようにします。

これは **オプトイン** です。何も設定しなければ、サーバーは起動時に余計なことを何も
しません（再配布したデプロイで「起動したのに1分間 CPU が張り付く」という驚きがない）。
有効にするには、インデックスを指定します。

```dotenv
# .env
HOOPS_AI_WARMUP_INDEXES=tmcad
HOOPS_AI_WARMUP_MAX=1
```

map 段階は、Web クライアントが新規ページで最初に要求するのと同じ母集団を
ウォームします（パーツのみ、3次元、クラスタ色分け、PCA 投影、タグ込み、
クラスタ数はインデックスのパーツ数から決まる値）。マップはパラメータの組ごとに
キャッシュされるため、これ以外をウォームしてもどのリクエストも当たらない
キャッシュキーを作るだけだからです。クラスタリングは k-means と Ward の
**両方** をウォームします（それぞれ別のキャッシュキーであるため）。また、
Ward のツリーが降りる最も細かいクラスタ数は、インデックスのパーツ数から
自動で決まります（1クラスタあたり約32パーツ。tmcad の単体パーツ 7,928 個なら
256）。これは `/index/{name}/stats` が `default_clusters` としてクライアントに
渡す値と同じで、k-means の **Clusters (k)** と Ward の **Max clusters** の
両方の既定になります。つまり、ユーザーが開くものは必ずウォーム済みです。

`HOOPS_AI_WARMUP_STAGES` で作業を絞り込めます（`assembly`、`map`、`scs`、
`searcher`。未設定ならすべて）。4つのうちアセンブリマッチャが最も遅いため、
これが効きます。

```dotenv
# アセンブリ検索を使わない構成では、その段階を省く
HOOPS_AI_WARMUP_STAGES=map,scs,searcher
```

Web クライアントの Shape Map は、ウォーム済みのクラスタリングのいずれかで
開きます。クライアントは起動時に `/client-config` からウォーム済みの一覧を
取得し、記憶している前回の選択がその中にあればそれを使い、無ければ先頭の値へ
落とします。いずれの場合も、ユーザーがマップを開くとコールドビルドではなく
ウォーム済みのクラスタリングが使われます。記憶している選択がそのまま使われる
のは、何もウォームされていないとき（ウォームアップが無効、または段階に `map`
が含まれないとき）だけです。

ウォームアップはバックグラウンドスレッドで実行されるため、サーバーは **即座に
リクエストを受け付けます**。ウォームアップ完了前に到着したものは、従来どおりコールドで
処理されます。通常のリクエストと同じビルドロックを共有するため、並行リクエストが同じ
ものの2度目のビルドを引き起こすことは決してなく、失敗（例: 破損したインデックス）は
ログに記録されてスキップされ、サーバーを落とすことはありません。レガシー（スキーマ v1）
インデックスはサーチャーのみをウォームします。進捗は `[WARMUP]` の下でログに出ます。

```
[WARMUP] warming 1 index(es): tmcad
[WARMUP] 'tmcad' searcher ready in 0.31s
[WARMUP] 'tmcad' shape map (kmeans) ready in 14.8s
[WARMUP] 'tmcad' assembly matcher ready in 21.2s
[WARMUP] complete in 36.4s
```

---

## ライセンス

このリポジトリ内の code は MIT License の下で公開されています。全文は [`LICENSE`](LICENSE) を参照してください。

MIT License が対象とするのは、このリポジトリの Contributors が書いた code のみです。third-party または HOOPS SDK component を再 license するものではありません。同梱または依存する third-party components、および HOOPS SDK components (HOOPS AI、HOOPS Exchange、HOOPS Visualize Web（そのオーサリングライブラリを含む)) は、それぞれ独自の terms の下に残ります。それらは [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md) と [`third_party_notices/`](third_party_notices/) に記録されています。

HOOPS SDK components を再配布できるか、またどの形式で再配布できるかは、ご自身の Technology Partner Agreement に依存します。この README 冒頭の *Status: sample / proof of concept* section を参照してください。
