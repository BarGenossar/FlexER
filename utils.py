

def create_multi_dict(path, intents_num):
    multi_label_dict = dict()
    for file_type in ['train', 'valid', 'test']:
        multi_label_dict[file_type] = dict()
        for intent in range(intents_num):
            current_file = open(path + file_type + str(intent) + '.txt', 'r', encoding="utf-8")
            Lines = current_file.readlines()
            for i, line in enumerate(Lines):
                if intent == 0:
                    multi_label_dict[file_type][i] = line[-2]
                else:
                    multi_label_dict[file_type][i] += line[-2]
            current_file.close()
    return multi_label_dict


def generate_txt_files(path, multi_label_dict):
    for file_type in ['train', 'valid', 'test']:
        current_file = open(path + file_type + '0.txt', "r", encoding="utf-8")
        Lines = current_file.readlines()
        new_file = open(path + file_type + '_Multilabel.txt', "w", encoding="utf-8")
        for i, line in enumerate(Lines):
            new_line = line.replace(line[-2], multi_label_dict[file_type][i])
            new_file.write(new_line)
        new_file.close()
    return
