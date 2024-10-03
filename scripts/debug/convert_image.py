from PIL import Image

# 打开PNG图片
png_image = Image.open('experiments/paper_man.png')

# 将图片转换为RGB模式
rgb_image = png_image.convert('RGB')

# 保存为JPG格式
rgb_image.save('experiments/paper_man.jpg')