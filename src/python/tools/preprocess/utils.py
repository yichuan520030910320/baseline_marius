from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve
import tarfile
import zipfile
from zipfile import ZipFile
import gzip
import shutil
import os
import argparse
import numpy as np
import torch
import os.path as osp
import os
from offgs.dataset import OffgsDataset

import dgl
from dgl.data import DGLDataset
import warnings
import contextlib
warnings.filterwarnings("ignore")
import time
class IGB260M(object):
    def __init__(
        self, root: str, size: str, in_memory: int, classes: int, synthetic: int
    ):
        self.dir = root
        self.size = size
        self.synthetic = synthetic
        self.in_memory = in_memory
        self.num_classes = classes

    def num_nodes(self):
        if self.size == "experimental":
            return 100000
        elif self.size == "small":
            return 1000000
        elif self.size == "medium":
            return 10000000
        elif self.size == "large":
            return 100000000
        elif self.size == "full":
            return 269346174

    @property
    def paper_feat(self) -> np.ndarray:
        num_nodes = self.num_nodes()
        # TODO: temp for bafs. large and full special case
        if self.size == "large" or self.size == "full":
            path = osp.join(self.dir, "full", "processed", "paper", "node_feat_128.npy")
            emb = np.memmap(path, dtype="float32", mode="r", shape=(num_nodes, 128))
        else:
            path = osp.join(self.dir, self.size, "processed", "paper", "node_feat.npy")
            if self.synthetic:
                emb = np.random.rand(num_nodes, 1024).astype("f")
            else:
                if self.in_memory:
                    emb = np.load(path)
                else:
                    emb = np.load(path, mmap_mode="r")

        return emb

    @property
    def paper_label(self) -> np.ndarray:
        if self.size == "large" or self.size == "full":
            num_nodes = self.num_nodes()
            if self.num_classes == 19:
                path = osp.join(
                    self.dir, "full", "processed", "paper", "node_label_19.npy"
                )
                node_labels = np.memmap(
                    path, dtype="float32", mode="r", shape=(num_nodes)
                )
                # Actual number 227130858
            else:
                path = osp.join(
                    self.dir, "full", "processed", "paper", "node_label_2K.npy"
                )
                node_labels = np.memmap(
                    path, dtype="float32", mode="r", shape=(num_nodes)
                )
                # Actual number 157675969

        else:
            if self.num_classes == 19:
                path = osp.join(
                    self.dir, self.size, "processed", "paper", "node_label_19.npy"
                )
            else:
                path = osp.join(
                    self.dir, self.size, "processed", "paper", "node_label_2K.npy"
                )
            if self.in_memory:
                node_labels = np.load(path)
            else:
                node_labels = np.load(path, mmap_mode="r")
        return node_labels

    @property
    def paper_edge(self) -> np.ndarray:
        path = osp.join(
            self.dir, self.size, "processed", "paper__cites__paper", "edge_index.npy"
        )
        # if self.size == 'full':
        #     path = '/mnt/nvme15/IGB260M_part_2/full/processed/paper__cites__paper/edge_index.npy'
        if self.in_memory:
            return np.load(path)
        else:
            return np.load(path, mmap_mode="r")

class IGB260MDGLDataset(DGLDataset):
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        super().__init__(name="IGB260MDGLDataset")

    def process(self):
        dataset = IGB260M(
            root=self.dir,
            size=self.args.dataset_size,
            in_memory=self.args.in_memory,
            classes=self.args.num_classes,
            synthetic=self.args.synthetic,
        )

        node_features = torch.from_numpy(dataset.paper_feat)
        node_edges = torch.from_numpy(dataset.paper_edge)
        node_labels = torch.from_numpy(dataset.paper_label).to(torch.long)

        self.graph = dgl.graph(
            (node_edges[:, 0], node_edges[:, 1]), num_nodes=node_features.shape[0]
        )
        self.graph.ndata["feat"] = node_features
        self.graph.ndata["label"] = node_labels

        self.graph = dgl.remove_self_loop(self.graph)
        self.graph = dgl.add_self_loop(self.graph)

        if self.args.dataset_size == "full":
            # TODO: Put this is a meta.pt file
            if self.args.num_classes == 19:
                n_labeled_idx = 227130858
            else:
                n_labeled_idx = 157675969

            n_nodes = node_features.shape[0]
            n_train = int(n_labeled_idx * 0.6)
            n_val = int(n_labeled_idx * 0.2)

            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)

            train_mask[:n_train] = True
            val_mask[n_train : n_train + n_val] = True
            test_mask[n_train + n_val : n_labeled_idx] = True

            self.graph.ndata["train_mask"] = train_mask
            self.graph.ndata["val_mask"] = val_mask
            self.graph.ndata["test_mask"] = test_mask
        else:
            n_nodes = node_features.shape[0]
            n_train = int(n_nodes * 0.6)
            n_val = int(n_nodes * 0.2)

            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)

            train_mask[:n_train] = True
            val_mask[n_train : n_train + n_val] = True
            test_mask[n_train + n_val :] = True

            self.graph.ndata["train_mask"] = train_mask
            self.graph.ndata["val_mask"] = val_mask
            self.graph.ndata["test_mask"] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return len(self.graphs)


def download_url(url, output_dir, overwrite):
    output_dir = Path(output_dir)

    url_components = urlparse(url)
    filename = Path(url_components.path + url_components.query).name
    filepath = output_dir / filename

    if filepath.is_file() and not overwrite:
        print(f"File already exists: {filepath}")
    else:
        try:
            print(f"Downloading {filename} to {filepath}")
            urlretrieve(url, str(filepath))
        except OSError:
            raise RuntimeError(f"Failed to download {filename}")

    return filepath


def extract_file(filepath, remove_input=True):
    try:
        if tarfile.is_tarfile(str(filepath)):
            if (str(filepath).endswith(".gzip") or
                    str(filepath).endswith(".gz")):
                with tarfile.open(filepath, "r:gz") as tar:
                    tar.extractall(path=filepath.parent)
            elif (str(filepath).endswith(".tar.gz") or
                  str(filepath).endswith(".tgz")):
                with tarfile.open(filepath, "r:gz") as tar:
                    tar.extractall(path=filepath.parent)
            elif str(filepath).endswith(".tar"):
                with tarfile.open(filepath, "r:") as tar:
                    tar.extractall(path=filepath.parent)
            elif str(filepath).endswith(".bz2"):
                with tarfile.open(filepath, "r:bz2") as tar:
                    tar.extractall(path=filepath.parent)
            else:
                try:
                    with tarfile.open(filepath, "r:gz") as tar:
                        tar.extractall(path=filepath.parent)
                except tarfile.TarError:
                    raise RuntimeError(
                        "Unrecognized file format, may need to perform extraction manually with a custom dataset.")
        elif zipfile.is_zipfile(str(filepath)):
            with ZipFile(filepath, "r") as zip:
                zip.extractall(filepath.parent)
        else:
            try:
                with filepath.with_suffix("").open("wb") as output_f, \
                        gzip.GzipFile(filepath) as gzip_f:
                    shutil.copyfileobj(gzip_f, output_f)
            except gzip.BadGzipFile:
                raise RuntimeError("Undefined file format.")
            except:
                raise RuntimeError("Undefined exception.")
    except EOFError:
        raise RuntimeError("Dataset file isn't complete. Try downloading again.")

    if filepath.exists() and remove_input:
        filepath.unlink()

    return filepath.parent


def strip_header(filepath, num_lines):
    cmd = "tail -n +{} {} > tmp.txt".format(num_lines+1, filepath)
    os.system(cmd)

    cmd = "mv tmp.txt {}".format(filepath)
    os.system(cmd)

   
import torch
import dgl
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.lsc import MAG240MDataset
import os


def load_ogb(name, root):
    data = DglNodePropPredDataset(name=name, root=root)
    splitted_idx = data.get_idx_split()
    g, labels = data[0]
    g: dgl.DGLGraph = g.long()
    feat = g.ndata["feat"]
    labels = labels[:, 0]
    if name == "ogbn-papers100M":
        labels[labels.isnan()] = 404.0
        labels = labels.long()
    n_classes = len(torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g.ndata.clear()
    g.edata.clear()
    return g, feat, labels, n_classes, splitted_idx


def load_igb(args):
    data = IGB260MDGLDataset(args)
    g: dgl.DGLGraph = data[0].long()
    n_classes = args.num_classes
    train_nid = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata["test_mask"], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata["val_mask"], as_tuple=True)[0]
    splitted_idx = {"train": train_nid, "test": test_nid, "valid": val_nid}
    feat = g.ndata["feat"]
    labels = g.ndata["label"]
    g.ndata.clear()
    g.edata.clear()
    return g, feat, labels, n_classes, splitted_idx


def load_mag240m(root: str, only_graph=True):
    dataset = MAG240MDataset(root=root)
    paper_offset = dataset.num_authors + dataset.num_institutions
    num_nodes = paper_offset + dataset.num_papers
    num_features = dataset.num_paper_features
    (g,), _ = dgl.load_graphs(os.path.join(root, "graph.dgl"))
    g: dgl.DGLGraph = g.long()
    train_idx = torch.LongTensor(dataset.get_idx_split("train")) + paper_offset
    valid_idx = torch.LongTensor(dataset.get_idx_split("valid")) + paper_offset
    test_idx = torch.LongTensor(dataset.get_idx_split("test-dev")) + paper_offset
    splitted_idx = {"train": train_idx, "test": test_idx, "valid": valid_idx}
    g.ndata.clear()
    g.edata.clear()
    feats, label = None, None
    if not only_graph:
        label = torch.from_numpy(dataset.paper_label)
        feats = torch.from_numpy(
            np.fromfile(os.path.join(root, "full_128.npy"), dtype=np.float32).reshape(
                num_nodes, 128
            )
        )
    return g, feats, label, dataset.num_classes, splitted_idx, paper_offset


def load_friendster(root: str, feature_dim: int, num_classes):
    graph_path = os.path.join(root, "friendster.bin")
    data, _ = dgl.load_graphs(graph_path)
    g: dgl.DGLGraph = data[0].long()
    # train_nid = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0]
    train_nid = torch.load(os.path.join(root, "train_010.pt"))
    test_nid = torch.nonzero(g.ndata["test_mask"], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata["val_mask"], as_tuple=True)[0]
    splitted_idx = {"train": train_nid, "test": test_nid, "valid": val_nid}
    g.ndata.clear()
    g.edata.clear()
    feats, labels = None, None
    if feature_dim != 0:
        feats = torch.rand((g.num_nodes(), feature_dim), dtype=torch.float32)
        labels = torch.randint(0, num_classes, (g.num_nodes(),), dtype=torch.int64)
    return g, feats, labels, num_classes, splitted_idx


def load_dglgraph(root: str, feature_dim: int, num_classes):
    data, _ = dgl.load_graphs(root)
    g: dgl.DGLGraph = data[0].long()
    train_nid = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata["test_mask"], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata["val_mask"], as_tuple=True)[0]
    splitted_idx = {"train": train_nid, "test": test_nid, "valid": val_nid}
    g.ndata.clear()
    g.edata.clear()
    feats, labels = None, None
    if feature_dim != 0:
        feats = torch.rand((g.num_nodes(), feature_dim), dtype=torch.float32)
        labels = torch.randint(0, num_classes, (g.num_nodes(),), dtype=torch.int64)
    return g, feats, labels, num_classes, splitted_idx
