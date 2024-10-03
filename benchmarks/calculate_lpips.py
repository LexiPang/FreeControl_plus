import torch
import lpips
import os
from IPython import embed
import torch.nn.functional as F

use_gpu = True         # Whether to use GPU
spatial = False         # Return a spatial map of perceptual distance.

# Linearly calibrated models (LPIPS)
loss_fn = lpips.LPIPS(net='squeeze', spatial=spatial) # Can also set net = 'squeeze' or 'vgg'
# loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'

if(use_gpu):
	loss_fn.cuda()

base_dir = './benchmarks/generated_imgs'
moving_avg = 0
index = 0

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    
    if os.path.isdir(folder_path):
        # 获取文件夹中的所有图片文件
        images = [img for img in os.listdir(folder_path) if img.endswith(('png', 'jpg', 'jpeg'))]
        images.sort()  # 排序文件，以确保获取前两张
        
        img1_path = os.path.join(folder_path, images[0])  # 第一张
        img2_path = os.path.join(folder_path, images[1])  # 第二张
        
        # 加载并转换图片
        img1 = lpips.im2tensor(lpips.load_image(img1_path))
        img2 = lpips.im2tensor(lpips.load_image(img2_path))
        
        if(use_gpu):
            img1 = img1.cuda()
            img2 = img2.cuda()

        # 计算 LPIPS 距离
        print(f"calculating:{folder_path}")
        
        # 如果图片大小不匹配，通过线性插值补齐
        if (img1.shape[-2], img1.shape[-1]) != (512, 512):
            img1 = F.interpolate(img1, size=(512, 512), mode='bilinear', align_corners=False)
        if (img2.shape[-2], img2.shape[-1]) != (512, 512):
            img2 = F.interpolate(img2, size=(512, 512), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            distance = loss_fn.forward(img1, img2)
            
        index += 1
        moving_avg = (moving_avg * (index - 1) + distance) / index
        if not spatial:
            print('Distances: (%.3f)'%(distance))
        else:
            print('Distances: (%.3f, %.3f)'%(distance.mean()))           # The mean distance is approximately the same as the non-spatial distance
    
            # Visualize a spatially-varying distance map between ex_p0 and ex_ref
            import pylab
            pylab.imshow(distance[0,0,...].data.cpu().numpy())
            pylab.show()
            
print('Average LPIPS distance: %.3f'%(moving_avg))
        # print(f'Folder: {folder}, LPIPS Distance: {distance.item():.3f}')
# ex_p0 = lpips.im2tensor(lpips.load_image("./benchmarks/generated_imgs/a sketch of a penguin_a video-game of a penguin_canny/img_0.png"))
# ex_p1 = lpips.im2tensor(lpips.load_image("./benchmarks/generated_imgs/a sketch of a penguin_a video-game of a penguin_canny/img_1.png"))

# if(use_gpu):
# 	ex_p0 = ex_p0.cuda()
# 	ex_p1 = ex_p1.cuda()

# ex_d0 = loss_fn.forward(ex_p0,ex_p1)

# if not spatial:
#     print('Distances: (%.3f)'%(ex_d0))
# else:
#     print('Distances: (%.3f, %.3f)'%(ex_d0.mean()))           # The mean distance is approximately the same as the non-spatial distance
    
#     # Visualize a spatially-varying distance map between ex_p0 and ex_ref
#     import pylab
#     pylab.imshow(ex_d0[0,0,...].data.cpu().numpy())
#     pylab.show()