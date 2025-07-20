import os
import shutil

def clean_subdirs(directory):
    for root, dirs, files in os.walk(directory, topdown=False):  # 从最深目录开始处理
        if root == directory:  # 跳过当前目录（只处理子目录）
            continue
            
        input_com_path = os.path.join(root, "input.com")
        has_input_com = os.path.exists(input_com_path)
        
        # (1) 删除子目录中的所有非 input.com 文件
        for file in files:
            file_path = os.path.join(root, file)
            if file != "input.com":
                try:
                    os.remove(file_path)
                    print(f"删除文件: {file_path}")
                except Exception as e:
                    print(f"删除失败: {file_path} ({e})")
        
        # (2) 如果子目录中没有 input.com，直接删除整个目录（包括子目录）
        if not has_input_com:
            try:
                shutil.rmtree(root)  # 强制删除整个目录（即使非空）
                print(f"删除目录（无input.com）: {root}")
            except Exception as e:
                print(f"删除目录失败: {root} ({e})")
        else:
            print(f"保留目录（含input.com）: {root}")

if __name__ == "__main__":
    current_dir = os.getcwd()
    print(f"开始清理子目录（仅保留input.com）: {current_dir}")
    clean_subdirs(current_dir)
    print("清理完成！")