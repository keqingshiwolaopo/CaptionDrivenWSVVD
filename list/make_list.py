import os
import glob


def create_list(directory, output_file):
    try:
        npy_files = []
        for file in os.listdir(directory):
            if file.endswith('.npy'):
                file_path = os.path.join(directory, file)
                npy_files.append(file_path)
             
        # 按文件名的字典序对文件路径进行排序
        npy_files.sort()

        with open(output_file, 'w') as f:
            for file_path in npy_files:
                f.write(f'{file_path}\n')
        print(f'成功创建 {output_file} 文件，其中包含 {len(npy_files)} 个 .npy 文件的路径。')
    except Exception as e:
        print(f'发生错误: {e}')


if __name__ == "__main__":
    # 请替换为实际的目录路径
    npy_path = '../data/datasets/UCF/I3D/RGBTest'
    list_path = './ucf/rgb_test.list'
    create_list(npy_path, list_path)
    
