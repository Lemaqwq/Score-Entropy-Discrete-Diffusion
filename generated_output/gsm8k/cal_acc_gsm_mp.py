import jsonlines
import re
import matplotlib.pyplot as plt

def extract_patterns(string):
    anws = string.split('####')[-1]
    anws = anws.split('<|endoftext|>')[0]
    anws = anws.strip()
    if anws.isnumeric:
        return anws
    else:
        return None
    


def calculate_correct_rate(jsonl_file):
    total_lines = 0
    correct_lines = 0


    with open(jsonl_file, 'r') as file:
        for data in jsonlines.Reader(file):
            
            recover = data['recover']
            source = data['source']

            if '####' in recover:

                if 'Sep' in recover and 'Pad' in recover:
                    recover = recover.split('Pad')[0]
                    recover = recover.split('Sep')[1]
                    recover = recover.strip()
                    
                    ans = extract_patterns(recover)

            
                    if len(ans) != 0:
                        try: 
                            reference = source.split('Pad')[0]
                            reference = reference.split('Sep')[1]
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
jsonl_file_path = 'step_1.jsonl'
correct_rate = calculate_correct_rate(jsonl_file_path)
print(f"Correct rate: {correct_rate}%")

# steps = [1, 2, 4, 8, 16, 32, 64, 128]
# acc = []

# for step in steps:
#     jsonl_file_path = f'step_{step}.jsonl'
#     correct_rate = calculate_correct_rate(jsonl_file_path)
#     acc.append(correct_rate)

# print(acc)

# plt.plot(steps, acc, scaley=True)
# plt.xlabel('Steps')
# plt.ylabel('Accuracy')
# plt.xticks(steps)
# plt.savefig('acc.png')


