#!/usr/bin/env python3
"""
プリセットUI実装の検証スクリプト
settings_panel.pyにプリセット選択UIが正しく実装されているか確認します
"""

import sys
from pathlib import Path

def verify_preset_ui():
    """settings_panel.pyの内容を検証"""
    panel_file = Path("gui/settings_panel.py")

    if not panel_file.exists():
        print("❌ gui/settings_panel.py が見つかりません")
        return False

    content = panel_file.read_text(encoding='utf-8')

    checks = {
        "環境プリセットグループボックス": "環境プリセット" in content,
        "プリセットComboBox": "self._preset_combo = QComboBox()" in content,
        "プリセット選択肢": '"Outdoor (屋外・高品質)"' in content,
        "プリセット説明ラベル": "self._preset_desc = QLabel" in content,
        "_on_preset_changedメソッド": "def _on_preset_changed" in content,
        "ConfigManagerインポート": "from core.config_loader import ConfigManager" in content,
        "レイアウトへの追加": "layout.addWidget(grp_preset)" in content,
    }

    print("=" * 60)
    print("プリセットUI実装チェック")
    print("=" * 60)

    all_passed = True
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("✅ すべてのチェックに合格しました！")
        print("\n次のステップ:")
        print("1. Pythonキャッシュをクリア: find . -type d -name '__pycache__' -exec rm -rf {} +")
        print("2. アプリケーションを再起動: python main.py")
        print("3. 右側パネル最上部の「環境プリセット」を確認")
    else:
        print("❌ いくつかのチェックに失敗しました")

    return all_passed

if __name__ == "__main__":
    success = verify_preset_ui()
    sys.exit(0 if success else 1)
