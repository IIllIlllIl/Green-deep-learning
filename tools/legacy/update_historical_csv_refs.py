#!/usr/bin/env python3
"""
更新配置文件中的历史CSV引用

将所有settings配置文件中的historical_csvs从summary_all.csv改为raw_data.csv
"""

import json
from pathlib import Path

def update_config_file(config_path: Path) -> bool:
    """更新单个配置文件"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # 检查是否有historical_csvs字段
        has_historical_csvs = False
        updated = False

        # 处理顶层的historical_csvs
        if 'historical_csvs' in config:
            has_historical_csvs = True
            old_csvs = config['historical_csvs']
            new_csvs = []
            for csv_path in old_csvs:
                if 'summary_all.csv' in csv_path:
                    new_path = csv_path.replace('summary_all.csv', 'raw_data.csv')
                    new_csvs.append(new_path)
                    updated = True
                else:
                    new_csvs.append(csv_path)
            config['historical_csvs'] = new_csvs

        # 处理experiments列表中的historical_csvs
        if 'experiments' in config:
            for exp in config['experiments']:
                if 'historical_csvs' in exp:
                    has_historical_csvs = True
                    old_csvs = exp['historical_csvs']
                    new_csvs = []
                    for csv_path in old_csvs:
                        if 'summary_all.csv' in csv_path:
                            new_path = csv_path.replace('summary_all.csv', 'raw_data.csv')
                            new_csvs.append(new_path)
                            updated = True
                        else:
                            new_csvs.append(csv_path)
                    exp['historical_csvs'] = new_csvs

        # 如果更新了，保存文件
        if updated:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            return True

        return False

    except Exception as e:
        print(f"⚠️  Error processing {config_path}: {e}")
        return False

def main():
    """主函数"""
    settings_dir = Path('/home/green/energy_dl/nightly/settings')

    print("=" * 70)
    print("更新配置文件中的历史CSV引用")
    print("=" * 70)
    print(f"从: results/summary_all.csv")
    print(f"到:   data/raw_data.csv")
    print("=" * 70)

    # 查找所有JSON配置文件
    config_files = list(settings_dir.glob('*.json'))

    updated_count = 0
    unchanged_count = 0

    for config_file in sorted(config_files):
        if update_config_file(config_file):
            print(f"✓ {config_file.name}")
            updated_count += 1
        else:
            unchanged_count += 1

    print("\n" + "=" * 70)
    print(f"更新完成")
    print("=" * 70)
    print(f"  更新: {updated_count} 个文件")
    print(f"  未变: {unchanged_count} 个文件")
    print(f"  总计: {len(config_files)} 个文件")

if __name__ == '__main__':
    main()
