import os

def rename_files_and_dirs_recursive(directory):
    """
    递归遍历指定目录及其所有子目录，将文件和文件夹名中的 '微珠' 替换为 'bead'，
    '细胞' 替换为 'cell'。
    """
    try:
        # 先处理子目录，确保递归时不影响路径
        for root, dirs, files in os.walk(directory, topdown=False):
            # 重命名文件
            for filename in files:
                old_file_path = os.path.join(root, filename)
                new_filename = filename.replace("微珠", "bead").replace("细胞", "cell")
                new_file_path = os.path.join(root, new_filename)

                if old_file_path != new_file_path:
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed file: {old_file_path} -> {new_file_path}")

            # 重命名目录
            for dirname in dirs:
                old_dir_path = os.path.join(root, dirname)
                new_dirname = dirname.replace("微珠", "bead").replace("细胞", "cell")
                new_dir_path = os.path.join(root, new_dirname)

                if old_dir_path != new_dir_path:
                    os.rename(old_dir_path, new_dir_path)
                    print(f"Renamed directory: {old_dir_path} -> {new_dir_path}")

    except Exception as e:
        print(f"Error: {e}")

# 指定目标目录
target_directory = r"C:\Users\76365\Desktop\241029\20241128TB model 采集"  # 替换为你的目标目录路径
rename_files_and_dirs_recursive(target_directory)

