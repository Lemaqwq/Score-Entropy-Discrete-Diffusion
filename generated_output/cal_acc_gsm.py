import jsonlines
import re

def extract_patterns(string):
    pattern = r'<<(.*?)>>'
    matches = re.findall(pattern, string)
    return ['<<{}>>'.format(match) for match in matches]

def calculate_correct_rate(jsonl_file):
    total_lines = 0
    correct_lines = 0


    with open(jsonl_file, 'r') as file:
        for data in jsonlines.Reader(file):
            
            recover = data['recover'][0]
            source = data['source'][0]

            if 'Sep' in recover and 'Pad' in recover:
                recover = recover.split('Pad')[0]
                recover = recover.split('Sep')[1]
                recover = recover.strip()
                ans = extract_patterns(recover)

        
                if len(ans) != 0:
                    try: 
                        ans = ans[-1].split('=')[1]
                        ans = ans.split('>>')[0]
                        ans = ans.strip()
                        if ans.isnumeric:
                            reference = source.split('Pad')[0]
                            reference = reference.split('Sep')[1]
                            reference = reference.strip()
                            reference = extract_patterns(reference)
                            reference = reference[-1].split('=')[1]
                            reference = reference.split('>>')[0]
                            if reference.isnumeric:
                                if ans == reference:
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
jsonl_file_path = 'gsm8kstep_128.jsonl'
correct_rate = calculate_correct_rate(jsonl_file_path)
print(f"Correct rate: {correct_rate}%")