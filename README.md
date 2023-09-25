# G2-AIMD: An Efficient GPU Framework for Subgraph Search

## Application 1: Maximal Clique Enumeration

### Dataset Preparation

Dataset can be prepared by round-robin method, or by k-means. 

When round-robin method is used:
```
wget http://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz
gzip -d com-friendster.ungraph.txt.gz
cd kmeans
make
./preprocess -f ../com-friendster.ungraph.txt
./kcore -f ../com-friendster.ungraph.bin
```

When k-means method is used, we adopt the same k-means based vertex binning method in VSGM, please check [here](https://github.com/kygx-legend/vsgm) for more details.
```
wget http://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz
gzip -d com-friendster.ungraph.txt.gz
cd kmeans
make
./preprocess -f ../com-friendster.ungraph.txt
./kcore -f ../com-friendster.ungraph.bin
./kmeans -f ../com-friendster.ungraph.bin
./view_pack -gf ../com-friendster.ungraph.bin -pf ../com-friendster.ungraph.bin.kmeans.4 -t 4
```

### Compile
```
cd app_BK
make
```

### Execution
```
# this example is using 4 producers, 4 consumers, queue size 4, and sorting the source vertices
./bk -dg ../com-friendster.ungraph.bin -h 2 -pn 4 -cn 4 -t 4 -qs 4 -ss 1
```

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
