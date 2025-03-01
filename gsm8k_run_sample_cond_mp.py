import time
import os
import torch
import argparse

from load_model import load_model_local
from transformers import GPT2TokenizerFast
from torch.utils.data import DataLoader
from data import finetune_get_dataset
import sampling
import data
import json

END_OF_THOUGHT_TOKEN_ID = 4242
EOS_TOKEN_ID = 50256
PAD_TOKEN_ID = 50256
SEP_TOKEN_ID = 15886
PRINT_END_OF_THOUGHT_TOKEN = "####"
PRINT_EOS_TOKEN = "<|endoftext|>"
PRINT_PAD_TOKEN = "<|endoftext|>"
PRINT_SEP_TOKEN = "||"


def update_input(first_sample, input_mask, input_ids, block_size=128):
    seq_len = len(first_sample)
    len_x = seq_len - input_mask.sum(dim=1)

    indices = (input_ids == SEP_TOKEN_ID).nonzero()
    if len(indices) > 0:
        index = indices[-1]
        # Slice the tensor to remove the repeating elements after the first 0
        ans = input_ids[index.item()+1:]

    # Remove all paddings
    mask = (ans != PAD_TOKEN_ID)
    ans = ans[mask]
    

    # Remove all paddings
    mask = (first_sample != PAD_TOKEN_ID)
    first_sample = first_sample[mask]

    # Remove Sep
    mask = (first_sample != SEP_TOKEN_ID)
    first_sample = first_sample[mask]

    # Delete all occurrences of eos
    mask = (first_sample != EOS_TOKEN_ID)
    first_sample = first_sample[mask]

    first_sample = torch.cat((first_sample, torch.tensor([EOS_TOKEN_ID, SEP_TOKEN_ID]).to(torch.device('cuda'))))

    updated_input_mask = torch.ones((1, block_size), dtype=torch.int64)
    updated_input_mask[0, :len(first_sample)] = 0

    first_sample = torch.cat((first_sample, ans))

    


    # pad to the same length
    first_sample = torch.nn.functional.pad(first_sample, (0, block_size - len(first_sample)), 'constant', PAD_TOKEN_ID)

    return first_sample.reshape(1, block_size), updated_input_mask.to(torch.device('cuda'))


     # Remove all paddings
    # Find the first occurrence of 0 in the tensor
    # indices = (first_sample == 15744).nonzero()
    # if len(indices) > 0:
    #     index = indices[0]
    #     # Slice the tensor to remove the repeating elements after the first 0
    #     first_sample = first_sample[:index.item()]

def batch_update_input(tensor, input_mask):
    seq_len = tensor.shape[1]
    new_tensor = []
    new_mask = tensor.new_ones(tensor.shape, dtype=bool)

    # indices = (input_ids == SEP_TOKEN_ID).nonzero()
    # if len(indices) > 0:
    #     index = indices[-1]
    #     # Slice the tensor to remove the repeating elements after the first 0
    #     ans = input_ids[index.item()+1:]

    # Remove all paddings
    # mask = (input_ids != PAD_TOKEN_ID)
    # input_ids = input_ids[mask]
    
    # Remove all paddings
    # mask = (tensor != PAD_TOKEN_ID)
    # tensor = tensor[mask]

    # # Remove Sep
    # mask = (tensor != SEP_TOKEN_ID)
    # tensor = tensor[mask]

    for i, b in enumerate(tensor.tolist()):
        try:  # remove sep and pad
            sep_token_idx = b.index(SEP_TOKEN_ID)
            b = b[:sep_token_idx-1] + b[sep_token_idx+1:]
            
            pad_token_idx = b.index(PAD_TOKEN_ID)
            b = b[:pad_token_idx]
            
        except:
            pass
        b.append(EOS_TOKEN_ID)
        b.append(SEP_TOKEN_ID)
        new_tensor.append(torch.tensor(b, dtype=torch.int64))
        new_mask[i][:len(b)] = False
    dummy_seq = torch.tensor([0]*seq_len, dtype=torch.int64)  # add a dummy seq with length=seq_len
    new_tensor = torch.nn.utils.rnn.pad_sequence([dummy_seq]+new_tensor, batch_first=True, padding_value=PAD_TOKEN_ID)
    new_tensor = new_tensor[1:]  # drop the dummy sequence
    new_tensor = new_tensor.type_as(tensor)
    new_mask = new_mask.type_as(input_mask)


    # batch_size = tensor.shape[0]
    # cleaned_tensors = []
    
    # for i in range(batch_size):
    #     mask = (tensor[i] != PAD_TOKEN_ID) & (tensor[i] != SEP_TOKEN_ID)
    #     cleaned_tensors.append(tensor[i][mask])


    # first_sample = torch.cat((first_sample, torch.tensor([EOS_TOKEN_ID, SEP_TOKEN_ID]).to(torch.device('cuda'))))

    # updated_input_mask = torch.ones((1, block_size), dtype=torch.int64)
    # updated_input_mask[0, :len(first_sample)] = 0

    # first_sample = torch.cat((first_sample, ans))

    


    # pad to the same length
    # first_sample = torch.nn.functional.pad(first_sample, (0, block_size - len(first_sample)), 'constant', PAD_TOKEN_ID)

    # return first_sample.reshape(1, block_size), updated_input_mask.to(torch.device('cuda'))
    return new_tensor, new_mask


def generate_samples(model, graph, noise, args, device, curr_batch_sz, block_size=128, input_ids=None, input_mask=None):

    def proj_fun(x):
        x = torch.where(input_mask==0, input_ids, x)
        return x

    sampling_fn = sampling.get_dot_pc_sampler(
                graph, noise, (curr_batch_sz, block_size), 'analytic', args.steps, device=device, proj_fun=proj_fun
        )
    
    samples = proj_fun(sampling_fn(model))

    return samples




def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="louaaron/sedd-medium", type=str)
    parser.add_argument("--dataset", default="wikitext103", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--prefix", type=str, default="Hi, my name is")
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    block_size = 128

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    test_set = finetune_get_dataset(args.dataset, "test", tokenizer, multipass=False, hidden_thought=False)

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        sampler=None,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        persistent_workers=True,
    )

    # mprint(f"Length of datasets: {len(train_ds)}, {len(eval_ds)}")

    test_iter = iter(test_loader)

    # def proj_fun(x):
    #     x = torch.where(input_mask==0, input_ids, x)
    #     return x

    device = torch.device('cuda')
    model, graph, noise = load_model_local(args.model_path, device)

    output_dir = f"generated_output/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    run_time = []

        
    for batch in test_iter:
        start_time = time.time()
        input_ids = batch["input_ids"].to(device)
        current_seq = tokenizer.batch_decode(input_ids)
        input_mask = batch["input_mask"].to(device)
        batch_size = len(input_ids)

        


        cot_steps = 10
        unfinished = input_ids.new_ones(batch_size, dtype=bool)
        end = False
        for _ in range(cot_steps):
            curr_batch_sz = unfinished.sum().item()
            samples = generate_samples(model, graph, noise, args, device, curr_batch_sz, block_size, input_ids[unfinished], input_mask[unfinished])
            input_ids[unfinished] = samples
            for i, item in enumerate(input_ids):
                if unfinished[i] and END_OF_THOUGHT_TOKEN_ID in item: 
                    unfinished[i] = False
                    if all(~unfinished):
                        end = True
            if end: 
                break
            
            # for unfinished x, remove sep, add sep at the first pad position
            input_ids[unfinished], input_mask[unfinished] = batch_update_input(input_ids[unfinished], input_mask)   



        

        # end_thought = False
        # len_of_thought = 0
        # while not end_thought:

        #     samples = proj_fun(sampling_fn(model))
        #     text_samples = tokenizer.batch_decode(samples)

        #     first_sample = samples[0]
        #     first_text_sample = text_samples[0]
        #     if "####" in first_text_sample or len_of_thought > 10:
        #         end_thought = True
        #     else:
        #         input_ids, input_mask = update_input(first_sample, input_mask, input_ids[0])
        #         decoded_input_ids_tmp = tokenizer.batch_decode(input_ids)
        #         # print(decoded_input_ids_tmp)
        #         len_of_thought += 1



        run_time.append(time.time() - start_time)
        if len(run_time) % 10 == 0:
            print(f"Thoughput (it/sec): {len(run_time) / sum(run_time)}")

        text_samples = tokenizer.batch_decode(input_ids)
        fout = open(output_dir + f"/step_{args.steps}.jsonl", 'a')
        for i in range(batch_size):
            print(json.dumps({"recover": text_samples[i], "source": tokenizer.decode(batch["input_ids"][i])}), file=fout)

        # for i in range(curr_batch_sz):
        #     print(json.dumps({"recover": text_samples[i], "source": tokenizer.decode(input_ids[i])}), file=fout)

    print(f"Thoughput (it/sec): {len(run_time) / sum(run_time)}")
        
    print("### Done!")


if __name__=="__main__":
    main()