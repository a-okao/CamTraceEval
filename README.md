# カメラ＋マーカー手先軌道評価システム

Python + OpenCV でマーカー付き手先の軌道を評価し、理想軌道（直線100mm / 円半径40mm）との誤差や所要時間を計測するツールです。ライブビュー上で計測開始/終了をキー操作し、結果を CSV とプロットで保存します。

## 1. 初回セットアップ

1. 依存ライブラリをインストール
   ```bash
   pip install -r requirements.txt  # ない場合は下記を個別インストール
   pip install opencv-python numpy matplotlib pyyaml
   ```
2. `config.yaml` を実環境に合わせて編集
   - `camera.device_id`: 使用するカメラ番号
   - `camera.calibrate_key`: キャリブレーションモードに入るキー（デフォルト `c`）
   - `marker.hsv_ranges`: 緑色マーカー用のHSV範囲。別色にしたい場合はここを変更
   - `marker.hsv_lower/hsv_upper`: 単一区間で使いたい場合のしきい値
   - `calibration`: スケール/オフセット or ホモグラフィ行列
   - `calibration.two_point_length_mm`: キャリブレーションでクリックする2点間を何mmとみなすか（デフォルト 100mm）
   - `trajectory`: 直線始点/終点、円の中心/半径
   - `output.base_dir` や `output.label` など
3. カメラを接続し、マーカーが映る位置にセット

## 2. 使い方（計測フロー）

### 起動コマンド例
- 直線軌道モード: `python main.py --mode LINE --label before`
- 円軌道モード: `python main.py --mode CIRCLE --label before`
- 設定ファイルを変える場合: `python main.py --config your_config.yaml --mode LINE`

### ライブビュー操作
- `[s]`: 計測開始
- `[e]`: 計測終了（自動で保存）
- `[q]` または `ESC`: 中断終了
- 計測中でなくても検出したマーカー座標 (u,v) / (x,y) を表示。検出位置は緑の十字線で重畳描画。
 - `[c]`（デフォルト）: キャリブレーションモード。
   - LINE: マウス左クリックで2点を順に指定し、2点目で自動終了。1点目を原点、2点目を x=100mm（設定値）とみなし、以降の mm 座標を更新。クリックした2点を結ぶ直線をライブビューに表示。
   - CIRCLE: マウス左クリックで「円の中心 → 外形上の1点」を指定し、2点目で自動終了。クリックした2点から中心と半径を算出し、以降の mm 座標変換・誤差計算・軌跡描画に使用。ライブビューにキャリブ済みの円と中心を描画。
- 計測終了後、結果（RMSE/最大誤差/サンプル数/所要時間、出力ファイル名）を別ウィンドウに表示し、保存した軌跡・誤差プロット画像も別窓で開く。任意キーで全ウィンドウを閉じる。

### 出力
- CSV: `outputs/mode_<mode>_<label>_<timestamp>.csv`
- 軌跡プロット: `..._trajectory.png`
- 誤差プロット: `..._error.png`

## 3. 2回目以降の流れ

1. 前回の設定が有効なら `config.yaml` をそのまま使用。条件が変わる場合だけしきい値やキャリブを更新。
2. 必要なら `--label after` などラベルを変えて実行し、学習前後を区別。
3. 生成された CSV やプロットを比較し、誤差(RMSE/最大誤差)や所要時間を評価。

## 4. 参考情報
- 依存ライブラリ: `opencv-python`, `numpy`, `matplotlib`, `pyyaml`
- 実機テスト（カメラ接続）はユーザー側で実施してください。ライブビューが出ない場合は `camera.device_id` を変更して再試行してください。
