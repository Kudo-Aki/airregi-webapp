# Airレジ 売上分析・需要予測 Webアプリ

神社の売上データを分析し、需要予測と納品計画を立てるためのWebアプリケーションです。

## 機能

### 🔍 商品検索・売上表示
- あいまい検索（全角半角対応、部分一致）
- 複数商品の合算/個別表示切り替え
- 日別・期間集計
- グラフ表示

### 📈 需要予測
- Prophet による時系列予測
- カレンダー要因（六曜、開運日、祝日）を考慮
- 年間・月別予測
- 信頼区間表示

### 📦 納品計画
- 最適納品スケジュール提案
- 在庫シミュレーション
- 欠品警告
- CSV エクスポート

## セットアップ

### 1. リポジトリをクローン

```bash
git clone https://github.com/your-username/airregi-webapp.git
cd airregi-webapp
```

### 2. 依存パッケージをインストール

```bash
pip install -r requirements.txt
```

### 3. Google Sheets API の設定

1. Google Cloud Console でサービスアカウントを作成
2. Google Sheets API を有効化
3. サービスアカウントキー（JSON）をダウンロード
4. スプレッドシートをサービスアカウントに共有

### 4. 認証情報の設定

**ローカル実行の場合:**

```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
```

**Streamlit Cloud の場合:**

Settings > Secrets に以下を追加:

```toml
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
# ... 以下、JSONキーの内容を全て貼り付け
```

### 5. カレンダーデータの準備

```bash
cd scripts
python generate_calendar.py
```

生成された `calendar_data.csv` をスプレッドシートの `calendar_data` シートにインポート。

### 6. アプリを起動

```bash
streamlit run app.py
```

## デプロイ（Streamlit Cloud）

1. GitHubにリポジトリをプッシュ
2. [Streamlit Cloud](https://share.streamlit.io/) にアクセス
3. 「New app」からリポジトリを選択
4. Settings > Secrets に認証情報を追加
5. デプロイ完了！

## ファイル構成

```
airregi-webapp/
├── app.py                    # メインアプリ
├── config.py                 # 設定ファイル
├── requirements.txt          # 依存パッケージ
├── modules/
│   ├── __init__.py
│   ├── data_loader.py       # データ読み込み
│   ├── product_normalizer.py # 商品名正規化
│   └── demand_forecast.py   # 需要予測
├── scripts/
│   └── generate_calendar.py # カレンダーデータ生成
└── .streamlit/
    ├── config.toml          # Streamlit設定
    └── secrets.toml.example # シークレットテンプレート
```

## スプレッドシート構成

| シート名 | 内容 |
|----------|------|
| daily_sales_summary | 日別売上集計 |
| daily_item_sales | 商品別売上 |
| calendar_data | カレンダーデータ（六曜、開運日等） |

## 将来の拡張予定

- [ ] Vertex AI 連携（高度な予測モデル）
- [ ] 自然言語クエリ対応
- [ ] リアルタイムアラート
- [ ] 多店舗比較機能

## ライセンス

Private
