# 設定統合の修正計画

## 現状の問題

### 1. ファイル保存の競合
- `settings_dialog.py` と `settings_panel.py` が同じ `~/.360split/settings.json` に書き込む
- settings_panel は一部の設定のみ保存するため、settings_dialog の設定を上書きすると情報が欠落する

### 2. 同期機能の欠如
- settings_dialog を閉じても settings_panel が再読み込みしない
- 両方のプリセット選択が独立している

## 解決策

### オプション A: settings_panel の保存を無効化（推奨）

settings_panel は「ライブプレビュー」専用とし、永続的な保存は settings_dialog のみで行う。

**メリット:**
- シンプルで安全
- settings_dialog が設定の「正式な保存場所」として明確
- データ欠落の心配がない

**実装:**
```python
# settings_panel.py の save_settings() メソッドをコメントアウトまたは削除
# 代わりに settings_dialog 閉じた後に settings_panel.reload_from_file() を呼ぶ
```

### オプション B: 完全統合

両方が同じ設定マネージャークラスを共有し、変更を相互に同期する。

**メリット:**
- どちらからでも永続的に保存できる
- リアルタイムで同期

**デメリット:**
- 実装が複雑
- settings_panel に全ての設定UIを追加する必要がある

## 推奨実装手順

### ステップ1: settings_panel の保存を無効化

```python
# settings_panel.py
def save_settings(self):
    """
    ライブプレビュー用パネルは保存しない。
    永続的な設定保存は settings_dialog から行う。
    """
    logger.info("settings_panel は保存を行いません。settings_dialog を使用してください。")
    return
```

### ステップ2: settings_dialog 閉じた後の再読み込み

```python
# main_window.py の _open_settings_dialog() メソッド
def _open_settings_dialog(self):
    dialog = SettingsDialog(self)
    if dialog.exec():
        logger.info("設定ダイアログが適用されました")
        # 設定パネルを再読み込み
        if hasattr(self, '_settings_panel'):
            self._settings_panel._load_settings()
            self._settings_panel._update_ui_from_config()
```

### ステップ3: プリセット同期（オプション）

settings_dialog でプリセットを選んだら、settings_panel にも反映させる。

```python
# main_window.py
def _open_settings_dialog(self):
    dialog = SettingsDialog(self)
    if dialog.exec():
        logger.info("設定ダイアログが適用されました")
        # 設定を再読み込み
        self._settings_panel._load_settings()

        # プリセット選択を同期（settings.json にプリセットIDを保存する場合）
        if 'selected_preset' in loaded_settings:
            preset_index = ...
            self._settings_panel._preset_combo.setCurrentIndex(preset_index)
```

## 最終的な動作フロー

1. **settings_panel (右パネル):**
   - リアルタイムでパラメータを調整
   - Live Preview でキーフレーム判定を即座に更新
   - **ファイルには保存しない**（一時的な調整のみ）

2. **settings_dialog (モーダルダイアログ):**
   - 全ての設定を網羅
   - OKボタンで `~/.360split/settings.json` に保存
   - **永続的な設定変更はここでのみ行う**

3. **main_window:**
   - settings_dialog が閉じたら settings_panel を再読み込み
   - 両方が常に同じ設定を表示する

## チェックリスト

- [ ] settings_panel.save_settings() を無効化
- [ ] settings_panel に _update_ui_from_config() メソッドを追加
- [ ] main_window で settings_dialog 閉じた後に settings_panel を再読み込み
- [ ] settings.json にプリセットID保存機能を追加（オプション）
- [ ] 動作確認: ダイアログで設定変更 → パネルに反映されることを確認
