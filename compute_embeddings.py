import sys

sys.path.append('../../CodeBERT/UniXcoder')
# sys.path.append('/Users/vincentberaudier/WML/CodeBERT/UniXcoder')

import elasticsearch
import sys
import os
from unixcoder import UniXcoder
import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from datasets import Dataset
from tree_sitter import Language, Parser
import torch
import tqdm
import time

from ray import data


# The first Parquet files contains around 110K files. We'll just take a subset for test purpose.

# The REBUILD_EMBEDDINGS flag is used to re-populate the vector database without recomputing all the functions source code embeddings which is the most time consuming part.
REBUILD_EMBEDDINGS = True
# JAVA   = False
# PYTHON = False
# JAVASCRIPT    = False
# GO = False
# TYPESCRIPT = False
# C = False
# CPP = True

NB_FILES_TO_PROCESS = 1000 #1000 * 30

BATCH_SIZE = 3

output_parquet = {
    "python": f'func_train-00000-of-00206_first_{NB_FILES_TO_PROCESS}.parquet',
    # "java": f'func_train-00000-of-00285_first_{NB_FILES_TO_PROCESS}.parquet',
    # "javascript": f'func_train-00000-of-00499_first_{NB_FILES_TO_PROCESS}.parquet',
    # "go": f'func_train-00000-of-00115_first_{NB_FILES_TO_PROCESS}.parquet',
    # "c": f'func_train-00000-of-00257_first_{NB_FILES_TO_PROCESS}.parquet',
    # "cpp": f'func_train-00000-of-00214_first_{NB_FILES_TO_PROCESS}.parquet',
    # "typescript": f'func_train-00000-of-00139_first_{NB_FILES_TO_PROCESS}.parquet',
}
input_parquet = {
    "python": '/Work/GitHub_CodeAssistant/bigcode/the-stack/data/python/train-00000-of-00206.parquet',
    # "java": '/Users/vincentberaudier/Downloads/train-00000-of-00285.parquet',
    # "python": '/Users/vincentberaudier/Downloads/train-00000-of-00206.parquet',
    # "javascript": '/Users/vincentberaudier/Downloads/train-00000-of-00499.parquet',
    # "go": '/Users/vincentberaudier/Downloads/train-00000-of-00115.parquet',
    # "c": '/Users/vincentberaudier/Downloads/train-00000-of-00257.parquet',
    # "cpp": '/Users/vincentberaudier/Downloads/train-00000-of-00214.parquet',
    # "typescript": '/Users/vincentberaudier/Downloads/train-00000-of-00139.parquet',
}

if len(output_parquet) == 0:
    raise Exception("We got a problem...")
if torch.backends.mps.is_available():
    device = torch.device("mps")# torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UniXcoder("microsoft/unixcoder-base")
model.to(device)

model.eval()
print('model.training = %s' % model.training)
#
# The function below will compute the embedding of a code snippet.
# It returns the embedding as a 2D numpy array of shape: (N, 768)
#
shape_size = 768
def get_func_embedding(func_src_list : list[str]):
    tokens_ids = model.tokenize(func_src_list, max_length=512, mode="<encoder-only>", padding=True)
    source_ids = torch.tensor(tokens_ids).to(device)
    _, func_embedding = model(source_ids)
    func_embedding = func_embedding.detach().cpu().numpy()
    func_embedding = func_embedding / np.linalg.norm(func_embedding, axis=1).reshape(-1, 1)
    return func_embedding

languages = {
    c: Language("build/my-languages.so", c) for c in ["python"]
}

parser = Parser()
parser.set_language(list(languages.values())[0])
#java_parser.set_language(JAVA_LANGUAGE)

if REBUILD_EMBEDDINGS:
    dfs = {}
    for k, v in input_parquet.items():
        print("Starting {0}".format(k))
        df = pd.read_parquet(v, engine='pyarrow')

        df.max_stars_repo_licenses = df.max_stars_repo_licenses.astype(
            str)  # Fix issue with: "TypeError: unhashable type: ..."
        df.max_forks_repo_licenses = df.max_forks_repo_licenses.astype(str)

        print('\tdf.max_stars_repo_licenses.nunique() = %s' % df.max_stars_repo_licenses.nunique())
        print('\tdf.max_forks_repo_licenses.nunique() = %s' % df.max_stars_repo_licenses.nunique())
        print("\t{0}".format(df.shape))
        dfs[k] = df
        print("Ending {0}".format(k))


min_size = 10
filtering = {
    "c": ["function_definition"],
    "cpp": ["function_definition"],
    "go": ['method_declaration'],
    "java": ['function_definition','constructor_declaration', 'method_declaration','generator_function_declaration',],
    "python": ['function_definition',],
    "typescript": ['expression_statement', 'function_declaration', 'method_definition'],
    "javascript": ['expression_statement', 'function_declaration'],
            }

def get_all_top_level_functions(language, root_node, current_start_end_points=None):
    if current_start_end_points is None:
        current_start_end_points = []

    for child in root_node.children:
        if child.type in filtering[language]:
            xx = abs(child.start_point[0] - child.end_point[0])
            if xx > min_size:
                current_start_end_points.append((child.start_point, child.end_point))
        else:
            current_start_end_points = get_all_top_level_functions(language, child, current_start_end_points)
    return current_start_end_points


def get_func(lines_py_src, start_point, end_point):
    first_line_idx, end_line_idx = start_point[0], end_point[0]
    if first_line_idx != end_line_idx:
        first_line = ' '*start_point[1] + lines_py_src[first_line_idx][start_point[1]:]
        end_line = lines_py_src[end_line_idx][:end_point[1]+1]
    else:
        first_line = lines_py_src[first_line_idx][start_point[1]:end_point[1]+1]
        end_line = ''
    lines = [first_line] + lines_py_src[first_line_idx+1:end_line_idx] + [end_line]
    f = lambda A, n=50: [A[i:i+n] for i in range(0, len(A), n)]
    sub_lines = f(lines, 50)
    # if len(sub_lines[len(sub_lines)]) < min_size:
    #     sub_lines.pop()
    ret = ["\n".join(sub) for sub in sub_lines]

    return ret


def extract_all_functions_embeddings(language, df, get_embedding):
    total_nb_funcs = 0
    # First pass, we collect the code snippets in a df2 with 2 columns: file number, and snippets.
    # We compute embedding for all snippets with batching.
    # 2nd pass, we iterate on DF, to get the row copy, then slice df2 with index1 == file number to collect all coressponding snippets.
    #    add this to func_rows.

    all_functs = []
    sizes = [0] * min([len(df), NB_FILES_TO_PROCESS+1])
    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print('idx = %s \t- total_nb_funcs = %s' % (idx, len(all_functs)))
        py_src = row.content
        lines_py_src = py_src.split('\n')
        if len(lines_py_src) >= min_size:
            tree = parser.parse(bytes(py_src, 'utf-8'))
            root_node = tree.root_node
            try:
                start_end_points = get_all_top_level_functions(language, root_node)
                if len(start_end_points) != 0:
                    for start_point, end_point in start_end_points:
                        splits = get_func(lines_py_src, start_point, end_point)
                        all_functs.extend(splits)
                        sizes[idx] = sizes[idx] + len(splits)
            except RecursionError:
                print('Skipping file: %s from repo: %s' % (row.max_stars_repo_path, row.max_stars_repo_name))
            except IndexError:
                raise IndexError
                break
        else:
            print('Skip file: %s from repo: %s' % (row.max_stars_repo_path, row.max_stars_repo_name))

        if idx == NB_FILES_TO_PROCESS:
            break  # Process only the first N files
    print("Found {0}/{1} functions".format(len(all_functs), sum(sizes)))

    func_embeddings = [None] * len(all_functs)
    idx = 0

    start_time = time.time()
    for i in range(0, len(all_functs), BATCH_SIZE):
        embeddings = get_embedding(all_functs[i:i + BATCH_SIZE])
        for j in range(0, embeddings.shape[0]):
            func_embeddings[idx] = embeddings[j]
            idx += 1
            if idx % 500 == 0:
                print("Number of snippets: {0}".format(idx))
    print('ELAPSED TIME = %s' % (time.time() - start_time))
    print("Step2 done: {0}".format(len(func_embeddings)))


    used_embedding = 0
    func_rows = [None] * len(func_embeddings)

    for idx, row in df.iterrows():
        if idx == NB_FILES_TO_PROCESS:
            break
        nb_to_embed = sizes[idx]
        if nb_to_embed != 0:
            for i in range(nb_to_embed):
                func_row = row.copy()
                func_row.content = all_functs[used_embedding]
                embed = func_embeddings[used_embedding]
                func_row['embedding'] = embed
                func_rows[used_embedding] = func_row
                used_embedding += 1
                if used_embedding % 500 == 0:
                    print("Number of snippets: {0}".format(used_embedding))

    func_df = pd.concat(func_rows, axis=1).T.reset_index(names='origin_id')
    func_df = func_df.reset_index(names='id')
    return func_df

for k,v in dfs.items():
    cols = v.columns
    to_keep = ["max_stars_repo_path", "max_stars_repo_name", "max_stars_repo_licenses",  "content"]
    to_drop = [c for c in cols if c not in to_keep]
    v.drop(to_drop, axis=1, inplace=True)

func_dfs = {}
if REBUILD_EMBEDDINGS:
    for k,v in dfs.items():
        print("Starting {0}".format(k))
        parser.set_language(languages[k])
        func_df = extract_all_functions_embeddings(k, v, get_func_embedding)
        print("Extraction done for {0}".format(k))
        func_df.to_parquet(output_parquet[k], engine='pyarrow')
        func_dfs[k] = func_df
        print("Ending {0}".format(k))
else:
    for k,v in output_parquet.items():
        func_df = pd.read_parquet(output_parquet[k], engine='pyarrow')
        func_dfs[k] = func_df






