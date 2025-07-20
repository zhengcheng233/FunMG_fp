import os

def rename_gjf_to_com(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.gjf'):
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, 'input.com')
                
                # 处理可能存在的重名文件
                if os.path.exists(new_path):
                    print(f"警告：{new_path} 已存在，跳过重命名 {old_path}")
                    continue
                
                try:
                    os.rename(old_path, new_path)
                    print(f"已重命名: {old_path} -> {new_path}")
                except Exception as e:
                    print(f"重命名 {old_path} 失败: {e}")

if __name__ == "__main__":
    current_dir = os.getcwd()
    print(f"开始处理目录: {current_dir}")
    rename_gjf_to_com(current_dir)
    print("处理完成")