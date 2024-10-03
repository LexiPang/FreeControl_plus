import yaml
import os
import subprocess
import time
import shlex

if __name__ == "__main__":
    # 打开并读取YAML文件
    with open('./benchmarks/ImageNetR-TI2I/imnetr-ti2i-source-prompt.yaml', 'r') as file:
        dataset = yaml.safe_load(file)

    # 查看解析后的数据
    for item in dataset:
        img_path = f"./benchmarks{item.get('init_img', None)}"  # 获取初始图片路径
        source_prompt = item.get('source_prompt', None)  # 获取初始提示（如果存在）
        target_prompts = item.get('target_prompts', [])  # 获取目标提示列表
        for target_prompt in target_prompts:
            for condition in ["canny","hed","scribble","depth","normal"]:
                print(f"Init Image: {img_path}")
                print(f"Source Prompt: {source_prompt}")
                print(f"Target Prompt: {target_prompt}")
                print(f"condition: {condition}")
                source_object = source_prompt.split()[-1]
                target_object = target_prompt.split()[-1]
                modified_src_prompt = source_prompt.replace(" ", "_")
                modified_trg_prompt = target_prompt.replace(" ", "_")
                script_name = f"./benchmarks/scripts/gen_img/{modified_src_prompt}_{modified_trg_prompt}_{condition}.sh"
                os.makedirs(os.path.dirname(script_name), exist_ok=True)
                with open(script_name, "w") as file:
                    file.write("#!/bin/bash\n\n")
                    file.write(f"python gen.py \\\n")
                    file.write(f'  --cond_img_path "{img_path}" \\\n')
                    file.write(f'  --prompt "{target_prompt}" \\\n')
                    file.write(f'  --paired_objs "({source_object}; {target_object})" \\\n')
                    file.write(f'  --pca_basis_dropdown "{source_object}_step_200_sample_20_id_0" \\\n')
                    file.write(f'  --inversion_prompt "{source_prompt}" \\\n')
                    file.write(f'  --condition "{condition}" \\\n')
                    # file.write(f'  --hide_verbose \\\n')
                    # file.write(f'  --scale 8 \\\n')
                    # if restart_list is not None:
                    #     file.write(f'  --restart \\\n')
                    #     file.write(f'  --restart_list {restart_list_str} \\\n')
                    # if second_order_option:
                    #     file.write(f'  --second_order \\\n')

                os.chmod(script_name, 0o755)

                # run script
                print(f"Running script: {script_name}")
                subprocess.run([f"{script_name}"])
        
        
        
        # print(f"Init Image: {init_img}")
        # print(f"Source Prompt: {source_prompt}")
        # print("Target Prompts:")
        # for prompt in target_prompts:
        #     print(f"  - {prompt}")
        # print("\n")
