import torch

def convert_deepspeed_ckpt_to_normal(deepspeed_ckpt_path, normal_ckpt_path):
    # Load the DeepSpeed checkpoint
    deepspeed_checkpoint = torch.load(deepspeed_ckpt_path, map_location=torch.device('cpu'))

    # Extract the model state dictionary from the DeepSpeed checkpoint
    model_state_dict = {}
    for key, value in deepspeed_checkpoint.items():
        if key == 'module':
            model_state_dict['state_dict'] = value
        else:
            model_state_dict[key] = value

    # Save the model state dictionary as a normal checkpoint
    torch.save(model_state_dict, normal_ckpt_path)

# Usage example
if __name__ == '__main__':
    deepspeed_ckpt_path = '/sharedata/home/daihzh/protein/ESM4SL/output/lora/PC9/C1/fold_0/best_ckpts/epoch=16-auroc=0.75.ckpt/checkpoint/mp_rank_00_model_states.pt'  # 'mp_rank_00_model_states.pt'
    normal_ckpt_path = '/sharedata/home/daihzh/protein/ESM4SL/output/lora/PC9/C1/fold_0/best_ckpts/final_model.ckpt'
    convert_deepspeed_ckpt_to_normal(deepspeed_ckpt_path, normal_ckpt_path)
