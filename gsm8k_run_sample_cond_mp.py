from datetime import time
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

def update_input(first_sample, input_mask, input_ids):
    seq_len = len(first_sample)
    len_x = seq_len - input_mask.sum(dim=1)

    indices = (input_ids == 8621).nonzero()
    if len(indices) > 0:
        index = indices[-1]
        # Slice the tensor to remove the repeating elements after the first 0
        ans = input_ids[index.item()+1:]

    # Remove all paddings
    mask = (ans != 15744)
    ans = ans[mask]
    

    # Remove all paddings
    mask = (first_sample != 15744)
    first_sample = first_sample[mask]

    # Remove Sep
    mask = (first_sample != 8621)
    first_sample = first_sample[mask]

    # Delete all occurrences of eos
    mask = (first_sample != 50256)
    first_sample = first_sample[mask]

    first_sample = torch.cat((first_sample, torch.tensor([50256, 8621]).to(torch.device('cuda'))))

    updated_input_mask = torch.ones((1, 128), dtype=torch.int64)
    updated_input_mask[0, :len(first_sample)] = 0

    first_sample = torch.cat((first_sample, ans))

    


    # pad to the same length
    first_sample = torch.nn.functional.pad(first_sample, (0, 128 - len(first_sample)), 'constant', 15744)

    return first_sample.reshape(1, 128), updated_input_mask.to(torch.device('cuda'))


     # Remove all paddings
    # Find the first occurrence of 0 in the tensor
    # indices = (first_sample == 15744).nonzero()
    # if len(indices) > 0:
    #     index = indices[0]
    #     # Slice the tensor to remove the repeating elements after the first 0
    #     first_sample = first_sample[:index.item()]



    



    new_input_ids = first_sample[:len_x]


def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="louaaron/sedd-medium", type=str)
    parser.add_argument("--dataset", default="wikitext103", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--prefix", type=str, default="Hi, my name is")
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    test_set = finetune_get_dataset(args.dataset, "test")

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

    

    def proj_fun(x):
        x = torch.where(input_mask==0, input_ids, x)
        return x

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
        curr_batch_sz = len(input_ids)
        sampling_fn = sampling.get_dot_pc_sampler(
                graph, noise, (curr_batch_sz, 128), 'analytic', args.steps, device=device, proj_fun=proj_fun
            )

        end_thought = False
        len_of_thought = 0
        while not end_thought:

            samples = proj_fun(sampling_fn(model))
            text_samples = tokenizer.batch_decode(samples)

            first_sample = samples[0]
            first_text_sample = text_samples[0]
            if "####" in first_text_sample or len_of_thought > 10:
                end_thought = True
            else:
                input_ids, input_mask = update_input(first_sample, input_mask, input_ids[0])
                decoded_input_ids_tmp = tokenizer.batch_decode(input_ids)
                # print(decoded_input_ids_tmp)
                len_of_thought += 1



        run_time.append(time.time() - start_time)


        fout = open(output_dir + f"/step_{args.steps}.jsonl", 'a')
        print(json.dumps({"recover": first_text_sample, "source": tokenizer.decode(input_ids[0])}), file=fout)

        # for i in range(curr_batch_sz):
        #     print(json.dumps({"recover": text_samples[i], "source": tokenizer.decode(input_ids[i])}), file=fout)

    print(f"Thoughput (it/sec): {len(run_time) / sum(run_time)}")
        
    print("### Done!")


if __name__=="__main__":
    main()