# Spike Sorting Toolkit
スパイクソーティングのワークフローを Python スクリプトでまとめたリポジトリです。  
合成データの生成 → 可視化 → 多チャンネル波形のスパイクソート（GUI ランチャ含む）までを一通り実行できます。

## ファイル構成
- `spike_sort_pipeline.py`  
  - Jupyter Notebook のロジックを整理したメイン処理。RAW からチャンネルごとにソートし、図/ログ/HDF5 を出力します。
- `spike_sort_launcher_gui.pyw`  
  - Tkinter 製のランチャ。RAW ファイルや出力先、対象チャンネルを GUI で指定して `spike_sort_pipeline.run_spike_sorting` を呼び出します。
- `generate_synthetic_raw.py`  
  - 解析用の合成 RAW データを生成するスクリプト。
- `launch_synthetic_raw.pyw`  
  - 合成データを対話的に生成・保存する GUI。
- `plot_synthetic_raw.py`  
  - 合成 RAW の内容を可視化して確認するスクリプト。
- `synthetic_raw_guide.md`  
  - 合成データの作り方や設定メモ。

## 必要環境
- Python 3.10+（Anaconda 環境を推奨）
- 主要ライブラリ: `numpy`, `scipy`, `scikit-learn`, `hdbscan`, `matplotlib`, `h5py`

## 使い方
### 1. 合成データの生成（任意）

- CLI で生成:  
```python generate_synthetic_raw.py --output ./test.raw --channels 2 --duration 60```
- GUI で生成:  
```python launch_synthetic_raw.pyw```

### 2. 合成 RAW の確認
```python plot_synthetic_raw.py --input ./test.raw --channels 2```

### 3. スパイクソート（GUI ランチャ）
```python spike_sort_launcher_gui.pyw```

1. RAW ファイルを選択  
2. 出力フォルダを設定（未指定なら `temp_results/`）  
3. チャンネルを複数選択（未選択で既定 62 ch）  
4. 「スパイクソーティング開始」を押す

実行完了後、ログ・HDF5・図は `temp_results/<RAW名>_spike_sort_<timestamp>/` に保存され、GUI 上のログも更新されます。

### 4. スパイクソート（コードから呼ぶ場合）
```python
from spike_sort_pipeline import run_spike_sorting

result = run_spike_sorting(
    raw_path="path/to/test.raw",
    channels=["ch21", "ch31"],
    output_root="temp_results",
)
print(result["summary"])
```

サンプル出力
下記は GUI ランチャで得られる結果の例です。
