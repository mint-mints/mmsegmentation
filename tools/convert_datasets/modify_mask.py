from PIL import Image
import os

# 指定文件夹路径
INPUT_DIR = "../../data/crack_dataset/annotation"
OUTPUT_DIR = "../../data/crack_dataset/ann_dir"

def process():
    # 获取文件夹中所有图片的文件名
    file_names = [f for f in os.listdir(INPUT_DIR) if f.endswith('.png')]

    # 循环遍历每个图片文件，并进行处理
    for file_name in file_names:
        # 打开图片
        img_path = os.path.join(INPUT_DIR, file_name)
        img = Image.open(img_path)

        # 将像素值为255的像素替换为1
        img = img.point(lambda x: 1 if x == 255 else x)

        # 保存处理后的图片
        new_img_path = os.path.join(OUTPUT_DIR, file_name)
        img.save(new_img_path)
        print('save ann_img:' + new_img_path)


if __name__ == "__main__":
    process()