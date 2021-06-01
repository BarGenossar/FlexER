import pandas as pd
from itertools import combinations
import argparse


class WDCProcessor:
    def __init__(self, file_path):
        self.categories = ['cameras', 'computers', 'shoes', 'watches']
        self.file_types = ['train', 'valid', 'test']
        self.file_path = file_path
        self.dfs = dict()
        self.id2string_left = dict()
        self.string2id_left = dict()
        self.id2string_right = dict()
        self.string2id_right = dict()
        self.convert_json2csv()
        self.create_MFI_datasets()

    def convert_json2csv(self):
        for category in self.categories:
            current_file_path1 = self.file_path
            current_file_path2 = current_file_path1 + category + '/'
            r_id_left, r_id_right = 0, 0
            self.dfs[category] = dict()
            for file_type in self.file_types:
                if file_type == 'test':
                    current_file_path3 = current_file_path2 + 'test.txt'
                else:
                    current_file_path3 = current_file_path2 + file_type + '.txt.small'
                r_id_left, r_id_right = self.create_df_type(current_file_path3, current_file_path2,
                                                            file_type, r_id_left, r_id_right, category)
            self.create_data_table('tableA', current_file_path2, self.id2string_left, category)
            self.create_data_table('tableB', current_file_path2, self.id2string_right, category)
        return

    def create_df_type(self, file_path, current_file_path2,
                       file_type, r_id_left, r_id_right, category):
        f = open(file_path, "r", encoding="utf-8")
        Lines = f.readlines()
        type_dict = dict()
        for i, line in enumerate(Lines):
            r1, r2_label = line.split('COL title VAL')[1:]
            r1 = r1[3:-1]
            r2, label = r2_label[3:].split('\t')
            label = int(label[0])
            if r1 not in self.string2id_left.keys():
                self.string2id_left[r1] = r_id_left
                self.id2string_left[r_id_left] = r1
                r_id_left += 1
            if r2 not in self.string2id_right.keys():
                self.string2id_right[r2] = r_id_right
                self.id2string_right[r_id_right] = r2
                r_id_right += 1
            type_dict[i] = {'ltable_id': self.string2id_left[r1],
                            'rtable_id': self.string2id_right[r2],
                            'label': label}
        df_type = pd.DataFrame.from_dict(type_dict, orient="index")
        df_type.to_csv(current_file_path2 + file_type + '.csv')
        self.dfs[category][file_type] = current_file_path2 + file_type + '.csv'
        return r_id_left, r_id_right

    def create_data_table(self, table_side, file_path, id2string, category):
        table = pd.DataFrame.from_dict(id2string, orient="index")
        table.reset_index(inplace=True)
        table.columns = ['id', 'title']
        table.to_csv(file_path + table_side + '.csv')
        self.dfs[category][table_side] = file_path + table_side + '.csv'
        return

    def get_df(self, category, table_side):
        df = pd.read_csv(self.dfs[category][table_side])
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        return df

    def create_MFI_datasets(self):
        categories = ['cameras', 'computers', 'shoes', 'watches']
        table_sides = ['tableA', 'tableB']
        for categories_pair in combinations(categories, 2):
            for table_side in table_sides:
                df1 = self.get_df(categories_pair[0], table_side)
                df2 = self.get_df(categories_pair[1], table_side)
                concatenated = pd.concat(([df1, df2]))
                concatenated.reset_index(inplace=True)
                concatenated.drop(['index', 'id'], axis=1, inplace=True)
                concatenated['check'] = 1
                concatenated.to_csv(self.file_path + 'MFI_WDC/' + categories_pair[0] + '_' +
                                    categories_pair[1] + '_' + table_side[-1] + '.csv')
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Preparation")

    hp = parser.parse_args()

    if hp.task == 'Preparation':
        WDC_obj = WDCProcessor('data/wdc/')
    elif hp.task == 'Enrich':
        pass
    print("Done")
