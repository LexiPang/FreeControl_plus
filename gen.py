"""
Generate images using provided configuration.
"""

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import gradio_app

def parse_key_list_pair(pair):
    key, value = pair.split("=")
    key = float(key)  # 将键转换为 float 类型
    value = [float(v) for v in value.split(",")]  # 将值转换为 float 列表
    return key, value

def main(args):
    """
    Main function to generate images.

    Parameters:
    args (argparse.Namespace): Namespace object containing image generation parameters.
    """
    # Convert the namespace object to a dictionary
    args_dict = vars(args)
    args_dict['condition_image'] = Image.open(args_dict['cond_img_path'])
    # if is png, convert to jpg
    if args_dict['condition_image'].format == 'PNG':
        args_dict['condition_image'] = args_dict['condition_image'].convert('RGB')
    if args_dict['restart_list'] is not None:
        args_dict['restart_list'] = dict(args_dict['restart_list'])
    
    gradio_app.model_dict, gradio_app.pca_basis_dict = gradio_app.load_ckpt_pca_list()

    # Call the gradio_app.freecontrol_generate method to generate a list of images
    img_list, denoising_img_list = gradio_app.freecontrol_generate(**args_dict)
    
    # Debug: Ensure images are generated
    if not img_list or len(img_list) == 0:
        print("No images generated.")
        return

    # Display the generated images
    print(len(img_list))
    for idx, img in enumerate(img_list):
        plt.imshow(img)
        plt.axis('off')
        if args.restart:
            restart_str = "True"
        else:
            restart_str = "False"
        if args.second_order:
            sec_str = "True"
        else:
            sec_str = "False"
        slash_idx = args.cond_img_path.rfind("/")
        cond_img = args.cond_img_path[slash_idx + 1:]
        
        # output_dir = f"./generated_imgs/{cond_img}/{args.prompt}"
        # output_filename = f"SG_{args.pca_guidance_weight}_restart_{restart_str}_restart_list_{dict(args.restart_list)}_second_order_{sec_str}_w_{args.scale}_steps_{args.ddim_steps}_{idx}.png"
        output_dir = f"./benchmarks/generated_imgs/{args.inversion_prompt}_{args.prompt}_{args.condition}"
        output_filename = f"img_{idx}.png" # output path used to run benchmarks
        output_path = os.path.join(output_dir, output_filename)
        os.makedirs(output_dir, exist_ok=True)
        
        # print(f'image saving to ./generated_imgs/{cond_img}/{args.prompt}/SG_{args.pca_guidance_weight}_restart_{restart_str}_restart_list_{dict(args.restart_list)}_second_order_{sec_str}_w_{args.scale}_steps_{args.ddim_steps}_{idx}.png')
        # output_path = f"./generated_imgs/{cond_img}/{args.prompt}/SG_{args.pca_guidance_weight}_restart_{restart_str}_restart_list_{dict(args.restart_list)}_second_order_{sec_str}_w_{args.scale}_steps_{args.ddim_steps}_{idx}.png"
        print(f'image saving to ./benchmarks/generated_imgs/{args.inversion_prompt}_{args.prompt}_{args.condition}/img_{idx}.png')
        # plt.savefig(output_path)
        img_tensor = transforms.ToTensor()(img)
        save_image(img_tensor, output_path)
        
        if idx == 1: # Save imgs and txts needed for CLIP score calculation
            clip_img_output_path = f"./benchmarks/clip_imgs/{args.inversion_prompt}_{args.prompt}_{args.condition}.png"
            clip_txt_output_path = f"./benchmarks/clip_txts/{args.inversion_prompt}_{args.prompt}_{args.condition}.txt"
            save_image(img_tensor, clip_img_output_path)
            with open(clip_txt_output_path, 'w') as file:
                file.write(args.prompt)
    
    if args.visualize:
        for i, (timestep, img_list) in enumerate(denoising_img_list):
            for idx, img in enumerate(img_list):
                img = np.squeeze(img, axis=0)
                output_filename = f"index({i})_timestep({timestep}).png"
                output_dir = f"./visualized_trajectories/{cond_img}_{args.prompt}_SG_{args.pca_guidance_weight}_restart_{restart_str}_restart_list_{dict(args.restart_list)}_second_order_{sec_str}_w_{args.scale}_steps_{args.ddim_steps}_{idx}/"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, output_filename)
                print(f"img saved to {output_path}")
                img_tensor = transforms.ToTensor()(img)
                save_image(img_tensor, output_path)
            
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image generation script")
    
    # Add command line arguments
    parser.add_argument('--cond_img_path', type=str, required=True, help='Path to the condition image')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt for image generation')
    parser.add_argument('--scale', type=float, default=7.5, help='Guidance scale')
    parser.add_argument('--ddim_steps', type=int, default=200, help='DDIM steps')
    parser.add_argument('--sd_version', type=str, default="1.5", help='Stable Diffusion version')
    parser.add_argument('--model_ckpt', type=str, default="naive", help='Model checkpoint')
    parser.add_argument('--pca_guidance_steps', type=float, default=0.6, help='Number of PCA guidance steps')
    parser.add_argument('--pca_guidance_components', type=int, default=64, help='Number of PCA guidance components')
    parser.add_argument('--pca_guidance_weight', type=int, default=600, help='PCA guidance weight')
    parser.add_argument('--pca_guidance_normalized', type=bool, default=True, help='Whether PCA guidance is normalized')
    parser.add_argument('--pca_masked_tr', type=float, default=0.3, help='PCA masked transformation')
    parser.add_argument('--pca_guidance_penalty_factor', type=int, default=10, help='PCA guidance penalty factor')
    parser.add_argument('--pca_warm_up_step', type=float, default=0.05, help='Number of PCA warm-up steps')
    parser.add_argument('--pca_texture_reg_tr', type=float, default=0.5, help='PCA texture regularization transformation')
    parser.add_argument('--pca_texture_reg_factor', type=float, default=0.1, help='PCA texture regularization factor')
    parser.add_argument('--negative_prompt', type=str, default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality", help='Negative prompt')
    parser.add_argument('--seed', type=int, default=2028, help='Random seed')
    parser.add_argument('--paired_objs', type=str, required=True, help='Paired objects')
    parser.add_argument('--pca_basis_dropdown', type=str, required=True, help='PCA basis dropdown')
    parser.add_argument('--inversion_prompt', type=str, required=True, help='Inversion prompt')
    parser.add_argument('--condition', type=str, default='None', help='Condition type')
    parser.add_argument('--img_size', type=int, default=512, help='Image size')
    parser.add_argument('--restart', action='store_true', help='Whether to restart')
    parser.add_argument('--second_order', action='store_true', help='Apply second order correction')
    parser.add_argument('--restart_list', type=parse_key_list_pair, nargs='+', help='Dictionary containing restart parameters, entered in the form of key1=value1,value2... key2=value3,value4')
    parser.add_argument('--hide_verbose', action='store_false', help='Hide progress bar')
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize the denoising process')
    parser.add_argument('--save_interval', type=int, default=1, help='the frequency at which to save images in the denoising loop')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Call the main function
    main(args)