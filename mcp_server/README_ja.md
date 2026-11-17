<!-- A Japanese translation is maintained in README_ja.md. When you change either README, update the other so both languages stay in sync. -->

# mcp_server — 類似検索サーバー用 MCP サーバー

**Languages:** [English](README.md) | 日本語

Web アプリを操作しなくても、Claude Desktop から類似検索サーバーの機能（比較、
埋め込み、インデックス検索、タグ、登録）を使えるようにする FastMCP サーバーです。
類似検索サーバーが起動していなければ、この MCP サーバーが起動します。

独自の軽い環境（`mcp` と `httpx` のみ）で動き、HTTP で類似検索サーバーと通信します。
HOOPS AI は import せず、HOOPS AI の仮想環境には何も追加しません。

## セットアップ

```bash
cd mcp_server
uv sync
```

## Claude Desktop への登録

`claude_desktop_config.json`（Windows: `%APPDATA%\Claude\claude_desktop_config.json`）の
`mcpServers` に追加します。

```json
{
  "mcpServers": {
    "hoops-ai-similarity": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\path\\to\\hoops_ai_similarity_search_server\\mcp_server",
        "run",
        "server.py"
      ],
      "env": {
        "HOOPS_SIM_URL": "http://127.0.0.1:8000",
        "HOOPS_SIM_PYTHON": "C:\\path\\to\\hoops_ai_venv\\Scripts\\python.exe"
      }
    }
  }
}
```

| 変数 | 既定値 | 意味 |
|------|--------|------|
| `HOOPS_SIM_URL` | `http://127.0.0.1:8000` | 類似検索サーバーの URL |
| `HOOPS_SIM_REPO` | `mcp_server` の 1 つ上のフォルダ | `main.py` があるフォルダ |
| `HOOPS_SIM_PYTHON` | （なし） | HOOPS AI の venv の `python.exe`。未設定なら自動起動しません |
| `HOOPS_SIM_AUTOSTART` | `1` | `0` で自動起動を止めます |
| `HOOPS_SIM_STARTUP_TIMEOUT` | `180` | 起動したサーバーの応答を待つ秒数 |
| `HOOPS_SIM_WORK_DIR` | `mcp_server/work` | `make_transformed_copy` の出力先 |

ツールに渡すファイルのパスは、**類似検索サーバーが動いているマシン上のパス**です
（サーバー自身が読みます）。そのマシンで使うか、同じパスが見えるサーバーに対して使ってください。

## ツール

| 分類 | ツール | 呼ぶ API |
|------|--------|----------|
| サーバー | `server_status`、`start_server`、`stop_server` | `GET /similarity/index/list`（応答確認） |
| 検証 | `compare_parts(paths, timeout)` | `POST /files/upload-from-path`、`POST /similarity/compare` |
| | `embed_part(path, include_vector)` | `POST /files/upload-from-path`、`POST /similarity/embed` |
| | `make_transformed_copy(path, translate, rotate_deg, axis)` | なし（ローカルのファイル） |
| | `translation_stability(path, shifts, rotations_deg, timeout)` | `compare_parts` と同じ |
| 検索 | `list_indexes`、`index_stats(index)` | `GET /similarity/index/list`、`GET /similarity/index/{name}/stats` |
| | `search_similar(index, path または part_id, k, mode)` | `POST /similarity/index/{name}/search` または `/search-assembly` |
| | `list_parts`、`get_part`、`get_part_thumbnail` | `GET /similarity/index/{name}/parts`、`.../parts/{id}/thumbnail` |
| タグ | `list_tags`、`parts_by_tag` | `GET .../tags`、`GET .../parts?tags=` |
| | `set_part_tags`、`tag_parts`、`untag_parts` | `PUT .../parts/{id}/tags`、`POST` / `DELETE .../tags/{tag}/parts` |
| 登録 | `create_index(name)` | `POST /similarity/index/create` |
| | `register_files(index, paths, folder, recursive)` | `POST /files/upload-from-path`、`POST .../jobs/add` |
| | `job_status`、`job_report` | `GET .../jobs/{id}`、`GET .../jobs/{id}/report` |

取り消せない操作（タグの廃止、全タグの一括削除、インデックスやパーツの削除）は
意図的に用意していません。Web アプリで行ってください。

応答はどれも数 KB 以内です。一覧には上限、総件数、`has_more` が付きます。
1 回の呼び出しはおよそ 1 分以内に返ります。登録はジョブにして状態を確認する形で、
`compare_parts` と `translation_stability` は `timeout` を受け取り、超えたときは
途中までの結果に `timed_out` を付けて返します。

API の制約に関する注記:

- `embed_part` が返すのはモデル、次元、ボディ数です。embed エンドポイントには
  OBB（oriented bounding box）がありません。
- `get_part` と `list_parts(filename=...)` は、1 件取得や名前フィルターの API が
  ないため、一覧をページ送りして探します。
- `register_files` は各ファイルをパスでアップロードしてからジョブを投入します
  （1 回 500 件まで）。残りは続けて呼び出してください。

## 自動起動と `stop_server`

どのツールも、最初に `GET /similarity/index/list` を送ります。応答がなく、
`HOOPS_SIM_AUTOSTART` が有効で、`HOOPS_SIM_PYTHON` が設定され、URL がローカルなら、
`HOOPS_SIM_REPO` で `main.py --host 127.0.0.1 --port <ポート>` を起動し、応答を待ちます。
`main.py` を直接起動するので、Windows の電源スロットリング対象外の設定が
埋め込みワーカーにも効きます。

- Windows では `DETACHED_PROCESS`、`CREATE_NEW_PROCESS_GROUP`、`CREATE_NO_WINDOW` を付けて、
  Linux では `start_new_session` で起動します。MCP サーバーが終了しても動き続けます。
- 出力は `mcp_server/logs/server_<日時>.log` に、PID は `mcp_server/.state/server.json` に
  記録します。
- `stop_server` が止めるのは、ここに記録したサーバー（プロセスツリーごと）だけです。
  ほかの方法で起動したサーバーは止めず、その旨を返します。

`logs/`、`.state/`、`work/` は git の管理対象外です。

## テスト

```bash
python -m pytest mcp_server
```

HTTP はモックで、サーバーは起動しません。
