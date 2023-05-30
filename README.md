# G2-thinker: A Multi-GPU Parallel Framework for Efficient Subgraph Search in a Big Graph

## Application 1: Maximal


## Application 2: Subgraph Enumeration

### Dataset Preparation
We adopt the same k-means based vertex binning method in VSGM, please check [here](https://github.com/kygx-legend/vsgm) for more details.
```
wget http://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz
gzip -d com-friendster.ungraph.txt.gz
cd kmeans
make
./preprocess -f ../com-friendster.ungraph.txt
./kmeans -f ../com-friendster.ungraph.bin
./view_pack -gf ../com-friendster.ungraph.bin -pf ../com-friendster.ungraph.bin.kmeans.4 -t 4
```

### Compile
```
cd app_gmatch
make
```

### Execution
```
./run -dg ../com-friendster.ungraph.bin -q 0 # triangle (4173724142)
./run -dg ../com-friendster.ungraph.bin -q 24 # 4-clique (8963503263)
```