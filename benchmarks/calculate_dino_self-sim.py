import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
from Splice.models.extractor import VitExtractor
import os
from IPython import embed
from torchvision import transforms as T
import torch.nn.functional as F
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        input_img_1 = T.Compose([
            T.Resize(224),
            T.ToTensor()
        ])(img1).unsqueeze(0).to(device)
        input_img_2 = T.Compose([
            T.Resize(224),
            T.ToTensor()
        ])(img2).unsqueeze(0).to(device)
        
        dino_preprocess = T.Compose([
            # T.Resize(224),
            # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        vit_extractor = VitExtractor('dino_vitb8', device)
        
        # calculate the keys
        with torch.no_grad():
            keys_self_sim_1 = vit_extractor.get_keys_self_sim_from_input(dino_preprocess(input_img_1), layer_num=11)
            keys_self_sim_2 = vit_extractor.get_keys_self_sim_from_input(dino_preprocess(input_img_2), layer_num=11)
        
        print(f"calculating:{folder_path}")
            
        # distance = torch.norm((keys_self_sim_1-keys_self_sim_2), p='fro')
        distance = F.mse_loss(keys_self_sim_1, keys_self_sim_2)
        # distance = F.l1_loss(keys_self_sim_1, keys_self_sim_2)
        
        index += 1
        moving_avg = (moving_avg * (index - 1) + distance) / index
        
        print('DINO-ViT self-similarity distance: (%.3f)'%(distance))
            
print('Average self-similarity distance: %.3f'%(moving_avg))
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