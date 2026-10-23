<!-- An English version is maintained in umap.md. When you change either file, update the other so both languages stay in sync. -->

# 任意: UMAP レイアウト

[English](umap.md) | [日本語](umap_ja.md) — [README](../README_ja.md) に戻る。

Shape Map は、各部品の 2,048 次元の集約埋め込みを描画できるように 3 次元へ落とします。
既定ではこれを PCA で行います。`umap-learn` をインストールすると、UMAP も選べるように
なります。

**変わるのは点の置き場所だけです。** k-means のクラスタリングは投影ではなく 2,048 次元
のベクトルに対して走るため、どの部品がどのクラスタに属するか — そしてクラスタの番号、
色、所属数、凝集度 — はどちらのレイアウトでも同一です。UMAP が変えるのは点が描かれる
位置だけです。

**導入する**

`umap-learn` は numba / llvmlite / pynndescent / scikit-learn を一緒に連れてきます。
そしてインストール先は HOOPS AI の仮想環境 — SDK 自身の numpy / scipy / torch / faiss
が同居している環境 — です。これが安全かどうかを決めている制約は numba のもので、
0.68.0 は `numpy>=1.22,<2.6` を宣言しています。Python 3.12 で入手できる numpy は
すべてこの範囲に収まっており、その両端（検証に使った 1.26.4 と、最新の 2.5.3）の
クリーンな環境から dry run を実行すると、追加されるのは 10 パッケージで、numpy と
scipy は変更されません。つまり現行のバージョンでは、これはごく普通のインストールです。

以下の 4 ステップは、失敗が予想されるから置いているのではありません。唯一の失敗
パターン — pip が numba の条件を満たすために numpy を動かすと決め、torch と faiss まで
道連れにすること — が、起きている最中は目に見えず、後から戻すのが厄介だから置いて
います。ステップ 2 は*あなたの*マシンでの答えを数秒で見えるようにし、ステップ 1 は
それを戻せるようにします。答えがほぼ確実に「何も動かなかった」であっても、1 分の
価値はあります。

1. **いま何が入っているかを記録する。** ファイルはこのチェックアウトの*外*に
   書いてください。リポジトリの一部と誤認されないようにするためです。

   **Windows:**
   ```bat
   <Path\to\HOOPS_AI\install\dir>\.venv\Scripts\pip.exe freeze > %USERPROFILE%\sdk-packages-before-umap.txt
   ```

   **Linux:**
   ```bash
   /path/to/HOOPS_AI/install/dir/.venv/bin/pip freeze > ~/sdk-packages-before-umap.txt
   ```

2. **pip が何をするつもりかを、実行せずに尋ねる。** pip 22.2 以降が必要です。お使いの
   pip が `--dry-run` を知らなければ、このステップは飛ばして構いません — ステップ 1 が
   守ってくれます。"Would install" の一覧を読んでください。検証に使ったバージョンでは
   10 パッケージが並び、その中に numpy は含まれません。もしお使いのマシンで `numpy`、
   `scipy`、`torch` が*現れた*なら、中止してください。SDK が依存しているパッケージを
   動かすつもりだと pip が告げているのであり、それは SDK の numpy が numba の範囲の
   外にあるということです。マップのレイアウトはそれに見合いません — 入れずに
   おきましょう。

   **Windows:**
   ```bat
   <Path\to\HOOPS_AI\install\dir>\.venv\Scripts\pip.exe install --dry-run -r requirements-umap.txt
   ```

   **Linux:**
   ```bash
   /path/to/HOOPS_AI/install/dir/.venv/bin/pip install --dry-run -r requirements-umap.txt
   ```

3. **インストールする。**

   **Windows:**
   ```bat
   <Path\to\HOOPS_AI\install\dir>\.venv\Scripts\pip.exe install -r requirements-umap.txt
   ```

   **Linux:**
   ```bash
   /path/to/HOOPS_AI/install/dir/.venv/bin/pip install -r requirements-umap.txt
   ```

4. **インストールを信用する前に、SDK がまだ読み込めることを確認する。** これはサーバーが
   import するものを import し、numpy と scipy のバージョンを表示します。ステップ 1 の
   ファイルと見比べてください。

   **Windows:**
   ```bat
   <Path\to\HOOPS_AI\install\dir>\.venv\Scripts\python.exe -c "import numpy, scipy; from hoops_ai.ml import CADSearch; print('SDK ok -- numpy', numpy.__version__, 'scipy', scipy.__version__)"
   ```

   **Linux:**
   ```bash
   /path/to/HOOPS_AI/install/dir/.venv/bin/python -c "import numpy, scipy; from hoops_ai.ml import CADSearch; print('SDK ok -- numpy', numpy.__version__, 'scipy', scipy.__version__)"
   ```

**インストール後はサーバーを再起動してください。** 利用可否はパッケージを import できる
かどうかで決まり、それはプロセスの起動時に一度だけ確定します。

有効になったかは `GET /client-config` で確認できます。有効なら `umap_available: true`
を返します。入っていなければ `false` を返し、Web クライアントは **503** を返すだけの
選択肢を見せる代わりに、マップのコントロールからその選択肢自体を出しません。入っていれば
他のマップコントロールの横に **Layout** の切り替えが現れます。いずれの場合も PCA の
挙動はこれまでと完全に同じで、既にディスクにあるキャッシュ済みのマップと `.scs` は
すべて有効なままです。

**元に戻す**

ステップ 1 で記録したバージョンを復元してください。`pip uninstall umap-learn` だけでは
**足りません**。パッケージは消えますが、インストールが置き換えた numpy はそのまま残り、
SDK を壊しているのはまさにその部分だからです。

**Windows:**
```bat
<Path\to\HOOPS_AI\install\dir>\.venv\Scripts\pip.exe install -r %USERPROFILE%\sdk-packages-before-umap.txt
```

**Linux:**
```bash
/path/to/HOOPS_AI/install/dir/.venv/bin/pip install -r ~/sdk-packages-before-umap.txt
```

そのファイルの中に再インストールできない行があれば、pip がその行を教えてくれます。
残りを入れ直したうえで、その 1 件だけを個別に処理してください。

UMAP は無料ではありません。2,048 次元・7,928 部品のインデックスで、投影の計算に
**約 25 秒**かかります。結果はキャッシュされ、クラスタ数 (`k`) や Outliers のしきい値を
変えてもやり直しには**なりません** — それらはクラスタリングだけを変えるためです。
ドリルダウンは部分集合に対して投影をやり直すので、そこでは再計算になります。

調整用の環境変数が 3 つあります。`HOOPS_AI_SHAPE_MAP_UMAP_N_NEIGHBORS`（既定 `15`）は
近傍の大きさで、局所的な細部と大域的な形のどちらを優先するかを決めます。
`HOOPS_AI_SHAPE_MAP_UMAP_MIN_DIST`（既定 `0.1`）は点どうしがどこまで密に詰まってよいかを
決めます。`HOOPS_AI_SHAPE_MAP_UMAP_SEED`（既定 `42`）は `random_state` を固定するもので、
レイアウトの再現性を保証すると同時に、umap-learn をシングルスレッドで動かす原因にも
なります。

> **既定が PCA のままである理由。** PCA の配置は線形で決定的です。同じファイル集合からは
> 常に同じ絵が得られ、ファイルを足しても絵はわずかにしか動きません。UMAP は k 近傍
> グラフから配置を作るため、その保証がありません — ファイルを足すと配置全体が回転したり
> 組み直されたりしえます。クラスタの中身と番号は変わりません。変わるのはその見え方です。
