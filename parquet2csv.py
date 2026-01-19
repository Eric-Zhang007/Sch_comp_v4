import pandas as pd
from pathlib import Path
import argparse
import sys

def batch_convert(input_dir: str, output_dir: str = None):
    root = Path(input_dir)
    
    # 如果没有指定输出目录，默认在输入目录下创建一个 csv_export 文件夹
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = root / "csv_export"
    
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 找到所有 parquet 文件
    files = list(root.glob("*.parquet"))
    
    if not files:
        print(f"Warning: No .parquet files found in {root}")
        return

    print(f"Found {len(files)} parquet files. Converting to CSV...")
    print(f"Output directory: {out_path.resolve()}")
    
    for fp in sorted(files):
        try:
            # 1. 读取 Parquet
            df = pd.read_parquet(fp)
            
            # 2. 构建输出文件名
            csv_name = fp.stem + ".csv"
            save_file = out_path / csv_name
            
            # 3. 保存为 CSV
            # index=True 保留时间索引，header=True 保留列名
            df.to_csv(save_file, index=True, encoding='utf-8-sig')
            
            print(f" -> Converted: {fp.name}  >>>  {csv_name}")
            
        except Exception as e:
            print(f" [ERROR] Failed to convert {fp.name}: {e}")

    print("\nAll done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Parquet files to CSV")
    parser.add_argument("--in_dir", required=True, help="Folder containing .parquet files")
    parser.add_argument("--out_dir", help="Folder to save .csv files (optional)")
    
    args = parser.parse_args()
    
    batch_convert(args.in_dir, args.out_dir)