import pandas as pd
# import run
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
    print("please select organelle name")
    matadata_df = pd.read_csv(METADATA_CSV_PATH)
    all_org = set(matadata_df['StructureDisplayName'])
    print(all_org)
    org = input()
    print(org)

    # print_full(matadata_df)
    # print_full((matadata_df.head()))
    # ['StructureDisplayName']
    # ['ChannelNumberBrightfield']


if __name__ == '__main__':
    cmd_script()