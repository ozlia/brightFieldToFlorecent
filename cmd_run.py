import pandas as pd
# import utils
from pandas import DataFrame

METADATA_CSV_PATH = "/sise/assafzar-group/assafzar/fovs/metadata.csv"
# interpreter_path = /home/omertag/.conda/envs/my_env/bin/python - change your user !!

def print_full(df: DataFrame):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(df)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')



def cmd_script():
    load = None
    while load not in ["y", "Y", "n", "N"]:
        print("Do you need to load new images? [y/n]")
        load = input()
    if load in ["y", "Y"]:
        print("Please select organelle name for this list:")
        print("---------------")
        matadata_df = pd.read_csv(METADATA_CSV_PATH)
        all_org = list(set(matadata_df['StructureDisplayName']))
        all_org.remove("None")
        for i in range(0, len(all_org), 3):
            print(', '.join(all_org[i:i + 3]))
        print("---------------")
        org = input()
        while org not in all_org:
            print("please enter valid name from the list above:")
            org = input()
    print(" how many e")

    # print_full(matadata_df)
    # print_full((matadata_df.head()))
    # ['StructureDisplayName']
    # ['ChannelNumberBrightfield']


if __name__ == '__main__':
    cmd_script()
    # run.run()