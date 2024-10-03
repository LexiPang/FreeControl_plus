import os
import itertools
import subprocess

if __name__ == "__main__":
    # 定义相连的参数（不做笛卡尔积）
    basic_params = [
        {
            "cond_img_path": "experiments/conditions/Human pose/pose_walking_woman.png",
            "prompt": "A man, full body, in the streets",
            "paired_objs": "(woman; man)",
            "inversion_prompt": "A photo of a woman, walking",
            "pca_basis_dropdown": "man_step_200_sample_20_id_0",
        },
        {
            "cond_img_path": "experiments/conditions/Human pose/pose_running_man.jpg",
            "prompt": "A man, full body, outside",
            "paired_objs": "(man; man)",
            "inversion_prompt": "A photo of a man, running",
            "pca_basis_dropdown": "man_step_200_sample_20_id_0",
        }
    ]

    # params to enumerate
    structure_guidances = [250,500,750,1000]
    second_order_options = [True, False]
    restart_options = [True, False]
    restart_lists = [{0.1:[10,2,2]},{1:[10,2,2]},{0.1:[10,2,2],5:[10,2,7.5]}]

    enumerated_params = list(itertools.product(structure_guidances, second_order_options))

    param_combinations = []
    for basic_param in basic_params:
        for restart in restart_options:
            if restart:
                for enumerated_param in enumerated_params:
                    for restart_list in restart_lists:
                        param_combinations.append((basic_param, enumerated_param, restart_list))
            else:
                for enumerated_param in enumerated_params:
                    param_combinations.append((basic_param, enumerated_param, None))

    # create a directory to store the scripts
    script_dir = "./scripts/restart_experiment"
    os.makedirs(script_dir, exist_ok=True)

    for i, (basic_param, (structure_guidance, second_order_option), restart_list) in enumerate(param_combinations):
        # generate script name
        slash_idx = basic_param["cond_img_path"].rfind("/")
        cond_img = basic_param["cond_img_path"][slash_idx + 1:]
        if second_order_option:
            second_order_str = "True"
        else:
            second_order_str = "False"
        if restart_list is not None:
            script_name = f"{script_dir}/{cond_img}/{basic_param['prompt']}/SG_{structure_guidance}_restart_True_restart_list_{restart_list}_second_order_{second_order_str}.sh"
            # 生成 restart_list 参数字符串
            restart_list_str = " ".join([f"{k}={','.join(map(str, v))}" for k, v in restart_list.items()])
        else:
            script_name = f"{script_dir}/{cond_img}/{basic_param['prompt']}/SG_{structure_guidance}_restart_False_second_order_{second_order_str}.sh"
            restart_list_str = ""
            
        os.makedirs(os.path.dirname(script_name), exist_ok=True)
        # create .sh file
        with open(script_name, "w") as file:
            file.write("#!/bin/bash\n\n")
            file.write(f"python gen.py \\\n")
            file.write(f'  --cond_img_path "{basic_param["cond_img_path"]}" \\\n')
            file.write(f'  --prompt "{basic_param["prompt"]}" \\\n')
            file.write(f'  --paired_objs "{basic_param["paired_objs"]}" \\\n')
            file.write(f'  --pca_basis_dropdown "{basic_param["pca_basis_dropdown"]}" \\\n')
            file.write(f'  --pca_guidance_weight "{structure_guidance}" \\\n')
            file.write(f'  --inversion_prompt "{basic_param["inversion_prompt"]}" \\\n')
            # file.write(f'  --hide_verbose \\\n')
            file.write(f'  --scale 8 \\\n')
            if restart_list is not None:
                file.write(f'  --restart \\\n')
                file.write(f'  --restart_list {restart_list_str} \\\n')
            if second_order_option:
                file.write(f'  --second_order \\\n')

        os.chmod(script_name, 0o755)

        # run script
        print(f"Running script: {script_name}")
        subprocess.run([f"./{script_name}"])

    print("All scripts have been generated and executed.")