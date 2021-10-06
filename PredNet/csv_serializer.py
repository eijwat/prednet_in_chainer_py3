import os
import numpy as np
from chainer import serializers
import tqdm


def npz_to_csv(directory, npz_data):
    for k, v in tqdm.tqdm(npz_data.items()):
        full_dir = os.path.join(directory, k)
        try:
            os.makedirs(full_dir)
        except OSError:
            print("already exist %s" % full_dir)
        if v.ndim <= 2:
            np.savetxt(os.path.join(full_dir, "000.csv"), v, delimiter=",")
        elif v.ndim == 3:
            for i in range(v.shape[0]):
                np.savetxt(os.path.join(full_dir, "%03d.csv") % i, v[i, ...], delimiter=",")
        elif v.ndim == 4:
            for i in range(v.shape[0]):
                for j in range(v.shape[1]):
                    np.savetxt(os.path.join(full_dir, "%03d_%03d.csv" % (i, j)), v[i, j, ...], delimiter=",")
        else:
            raise ValueError("Cannot support %d-dimension tensor." % v.ndim)


def save_to_csv(directory, obj):
    s = serializers.DictionarySerializer()
    s.save(obj)
    target = s.target
    npz_to_csv(target)


def csv_to_npz(directory):
    dic_params = {}
    for curdir, dirs, files in tqdm.tqdm(list(os.walk(directory))):
        csv_files = []
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(file)
        if len(csv_files) > 0:
            key = curdir.lstrip(directory)
            if key[0] == '/':
                key = key[1:]
            name, ext = csv_files[0].split('.')
            fdim = len(name.split('_'))
            if fdim == 1 and len(csv_files) == 1:
                mat = np.loadtxt(os.path.join(curdir, csv_files[0]), delimiter=",", dtype=np.float32)
            elif fdim == 1 and len(csv_files) > 1:
                mat = []
                for f in sorted(csv_files):
                    mat.append(np.loadtxt(os.path.join(curdir, f), delimiter=",", dtype=np.float32))
                mat = np.stack(mat)
            elif fdim == 2:
                mat = []
                rows = 0
                cols = 0
                for f in sorted(csv_files):
                    name, ext = f.split('.')
                    r, c = name.split('_')
                    r, c = int(r), int(c)
                    if r > rows:
                        rows = r
                    if c > cols:
                        cols = c
                    mat.append(np.loadtxt(os.path.join(curdir, f), delimiter=",", dtype=np.float32))
                mat = np.stack(mat)
                mat = mat.reshape((rows + 1, cols + 1, mat.shape[1], mat.shape[2]))
            dic_params[key] = mat
    return dic_params


def load_from_csv(directory, model):
    dic_params = csv_to_npz(directory)
    d = serializers.NpzDeserializer(dic_params)
    d.load(model)

"""
def check_converter(npz_data=np.load("models/initial.model")):
    print("Start checking...")
    npz_to_csv("test", npz_data)
    params = csv_to_npz("test")
    for k, v in npz_data.items():
        if k in params:
            res = np.allclose(params[k], v)
            print(k, res, params[k].shape, v.shape)
            if not res:
                print(params[k], v)
        else:
            raise ValueError("Fail checking.")
"""

if __name__ == "__main__":
    # check_converter()
    import argparse
    parser = argparse.ArgumentParser(description='csv_serializer')
    subparsers = parser.add_subparsers(dest='command')
    parser_to_csv = subparsers.add_parser('npz_to_csv', help='see `add -h`')
    parser_to_csv.add_argument('input', type=str, help='Path to npz file')
    parser_to_csv.add_argument('--directory', '-dir', default='test', type=str, help='Path to directory to save csv files')

    parser_to_npz = subparsers.add_parser('csv_to_npz', help='see `commit -h`')
    parser_to_npz.add_argument('output', type=str, help='Path to output npz file')
    parser_to_npz.add_argument('--directory', '-dir', default='test', type=str, help='Path to directory to load csv files')
    args = parser.parse_args()
    if args.command == 'npz_to_csv':
        npz_to_csv(args.directory, np.load(args.input))
    else:
        params = csv_to_npz(args.directory)
        with open(args.output, 'wb') as f:
            np.savez_compressed(f, **params)
