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

        
    for batch in test_iter:
        input_ids = batch["input_ids"].to(device)
        input_mask = batch["input_mask"].to(device)
        curr_batch_sz = len(input_ids)



        sampling_fn = sampling.get_dot_pc_sampler(
            graph, noise, (curr_batch_sz, 128), 'analytic', args.steps, device=device, proj_fun=proj_fun
        )

        samples = proj_fun(sampling_fn(model))


        text_samples = tokenizer.batch_decode(samples)


        fout = open(output_dir + f"/step_{args.steps}.jsonl", 'a')

        for i in range(curr_batch_sz):
            print(json.dumps({"recover": text_samples[i], "source": tokenizer.decode(input_ids[i])}), file=fout)
        
    print("### Done!")


if __name__=="__main__":
    main()