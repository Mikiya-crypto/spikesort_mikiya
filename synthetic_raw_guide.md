# 合成RAWデータ生成と確認の手順

## 概要
- 実験用の擬似生体信号を `generate_synthetic_raw.py` で生成し、GUI ランチャー `launch_synthetic_raw.pyw` から操作できます。
- 出力はヘッダ無しの int16 リトルエンディアン RAW で、チャンネルごとにデータがインターリーブされています。
- 波形の可視化およびチャンネル構成の確認には `plot_synthetic_raw.py` を使用します。

## データ生成ツール `launch_synthetic_raw.pyw`
- `SynthConfig` の既定値をフォームに読み込み、各パラメータを入力すると `generate_synthetic_raw.generate_synthetic_raw` を呼び出してデータを生成します。
- 主なパラメータ
  - Sampling rate `fs` [Hz]
  - Duration `duration_s` [s]
  - Channels `n_channels`
  - ノイズ、スパイク振幅・発生率、ユニット数、リフラクトリー時間、乱数シード
- 「Browse…」で保存先を指定。生成後は `save_raw_int16_le` で int16 RAW に変換し、必要に応じて保存フォルダを自動で開きます。
- GUI をダブルクリックで起動できるよう、スクリプト直下でプロジェクトルートを `sys.path` に追加しています。

## 生成ロジック `generate_synthetic_raw.py`
- `SynthConfig` dataclass がデフォルトの信号設定を保持します。
- `_make_spike_waveform` で 1.2 ms 相当の二相性スパイク波形を作成し、Poisson 過程 `_poisson_spike_train` でスパイクタイムを決定します。
- `generate_synthetic_raw` は雑音 + スパイクを合成した float32 アレイを返し、`save_raw_int16_le` で int16 RAW に変換します。

## データ確認ツール `plot_synthetic_raw.py`
- コマンド例: `python plot_synthetic_raw.py --seconds 0.05 --channels 0 1`
- 主な引数
  - `--path`: RAW ファイルパス (既定は `リポジトリ/synthetic.raw`)
  - `--fs`: サンプリング周波数
  - `--n-channels`: チャンネル数を直接指定 (省略時は推定)
  - `--duration-hint`: 推定時の録音時間の目安 [s]
  - `--channels`: 描画するチャネル番号 (省略時は全チャネル)
  - `--seconds` / `--offset`: 描画区間の長さと開始時刻
  - `--no-channel-gui`: チャンネル推定ダイアログを開かない
- RAW サイズを基に複数のチャンネル候補を算出し、Tkinter のダイアログでユーザーに確認してもらう仕組みを追加しました。GUI を無効化した場合は候補リストを標準出力に表示し、`--duration-hint` で意図した長さを優先できます。
- 推定結果や指定されたチャンネル数に基づき、Matplotlib のサブプロットで各チャネルの波形を描画します。

## 推奨ワークフロー
1. `launch_synthetic_raw.pyw` を実行し、生成したい条件に合わせてパラメータを設定して RAW を保存します。
2. 生成直後に表示されるダイアログでファイル名・チャンネル数・推定ファイルサイズを確認します。
3. 波形確認は `plot_synthetic_raw.py` を実行。チャンネル数が自動推定と異なる場合はダイアログで選択するか、`--n-channels` または `--duration-hint` を指定します。
4. `--seconds` と `--offset` で任意の時間帯を拡大し、各チャネルの波形を視覚的に検証します。

## 補足とトラブルシュート
- RAW サイズとサンプリング周波数が分かっていれば、`Get-Item <path> | Select-Object Name, Length` でおおよそのチャンネル数と録音時間を計算できます。
- 旧来の Python バージョン (3.9 以前) を使う場合は、`generate_synthetic_raw.py` 冒頭の `from __future__ import annotations` により `str | Path` などの型ヒントを使用可能にしています。
- GUI ダイアログが開かない環境では `--no-channel-gui` を付けたうえで `--n-channels` を明示してください。
