import jsonlines
import re
import matplotlib.pyplot as plt

def extract_patterns(string):
    ans = string.split('<|endoftext|>')[0]
    ans = ans.split('####')[-1]
    ans = ans.strip()
    if ans.isnumeric:
        return ans
    else:
        return None
    


def calculate_correct_rate(jsonl_file):
    total_lines = 0
    correct_lines = 0


    with open(jsonl_file, 'r') as file:
        for data in jsonlines.Reader(file):
            
            recover = data['recover']
            source = data['source']

            if '<|endoftext|>[SEP]' in recover:
                recover = recover.split('<|endoftext|>[SEP]')[1]
                if '<|endoftext|>' in recover:
                    recover = recover.strip()
                    ans = extract_patterns(recover)

                    if len(ans) != 0:
                        try: 
                            reference = source.split('[PAD]')[0]
                            reference = reference.split('[SEP]')[1]
                            reference = reference.strip()
                            reference = extract_patterns(reference)
                            if reference.isnumeric:
                                if ans == reference:
                                    print(total_lines)
                                    correct_lines += 1
                                    # print(f"reference: {data['source']}\nrecover: {data['recover']}")
                            else:
                                print(f"Bad reference detected: reference: {data['source']}\nrecover: {data['recover']}")
                        except:
                            print(f"Bad reference detected: reference: {data['source']}\nrecover: {data['recover']}")
                            continue
                    


            total_lines += 1
           

    correct_rate = correct_lines / total_lines * 100
    print(f'Correct lines: {correct_lines}\nTotal lines: {total_lines}')

    return correct_rate

# Usage example
# jsonl_file_path = 'step_64.jsonl'
# correct_rate = calculate_correct_rate(jsonl_file_path)
# print(f"Correct rate: {correct_rate}%")

steps = [4, 8, 16, 32, 64]
acc = []

for step in steps:
    jsonl_file_path = f'/workspace/generated_output/gsm8k/dot_medium/step_{step}.jsonl'
    correct_rate = calculate_correct_rate(jsonl_file_path)
    acc.append(correct_rate)

print(acc)

plt.plot(steps, acc, scaley=True)
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.xticks(steps)
plt.savefig('12000_small_acc.png')


