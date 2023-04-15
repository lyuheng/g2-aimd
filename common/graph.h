
#ifndef COMMON_GRPAH_H
#define COMMON_GRPAH_H

#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <set>
#include <queue>
#include <utility>
#include <fstream>
#include <string.h>

#include "common/meta.h"
#include "common/timer.h"
#include "bliss/graph.hh"


typedef std::pair<CondOperator, uintV> CondType;
typedef std::vector<std::vector<CondType>> AllCondType;

enum GraphType { QUERY, DATA }; 

enum PresetPatternType {
    P0,   // triangle, hold because enum begins with 0
    P1,   // square
    P2,   // chrodal square
    P3,   // 2 tails triangle
    P4,   // house
    P5,   // chrodal house
    P6,   // chrodal roof
    P7,   // three triangles
    P8,   // solar square
    P9,   // near 5 clique
    P10,  // four triangles
    P11,  // one in three triangles
    P12,  // near 6 clique
    P13,  // square on top
    P14,  // near 7 clique
    P15,  // 5 clique on top
    P16,  // 5 circles
    P17,  // 6 circles
    P18,  // hourglass

    P19,  // placeholder
    P20,  // placeholder
    P21,  // placeholder
    P22,  // placeholder
    P23,  // 3 clique
    P24,  // 4 clique
    P25,  // 5 clique
    P26,  // 6 clique
    P27   // 7 clique
};

class Graph
{
public:
    Graph() {}
    Graph(const std::string &filename, PresetPatternType queryType, GraphType gt);

    virtual ~Graph() {}

    size_t GetEdgeCount() const { return edge_count_; }
    size_t GetVertexCount() const { return vertex_count_; }
    void SetVertexCount(size_t vertex_count) { vertex_count_ = vertex_count; }
    void SetEdgeCount(size_t edge_count) { edge_count_ = edge_count; }

    uintE *GetRowPtrs() const { return row_ptrs_; }
    uintV *GetCols() const { return cols_; }

    void SetRowPtrs(uintE *row_ptrs) { row_ptrs_ = row_ptrs; }
    void SetCols(uintV *cols) { cols_ = cols; }

    void GetMaxDegree();

    void readBinFile(const std::string &filename);
    void readGraphFile(const std::string &filename);
    void readSnapFile(const std::string &filename);
    void readLVIDFile(const std::string &filename);

    void writeBinFile(const std::string &filename);

    
    bliss::Graph* GetBlissGraph();
    std::vector<std::pair<uintV, uintV>> GetConditions(bliss::Graph* bg);
    std::vector<std::vector<uintV>> GetAutomorphisms(bliss::Graph* bg);
    void CompleteAutomorphisms(std::vector<std::vector<uint32_t>>& perm_group);
    std::map<uintV, std::set<uintV>> GetAEquivalenceClasses(const std::vector<std::vector<uintV>>& aut);

    void SetConditions(const std::vector<std::pair<uintV, uintV>>& conditions);

    void Preprocess();

    void ReMapVertexId();

    std::vector<std::vector<uintV>> selectPresetPatterns(PresetPatternType patternType);

    AllCondType order_;

private:
    uintE *row_ptrs_;
    uintV *cols_;
    size_t vertex_count_;
    size_t edge_count_;
};

Graph::Graph(const std::string &filename, PresetPatternType queryType = P0, GraphType gt = GraphType::DATA)
{
    if (gt == GraphType::DATA)
    {
        std::string suffix = filename.substr(filename.rfind(".") + 1);
        if (suffix == "bin")
            readBinFile(filename);
        else if (suffix == "graph")
            readGraphFile(filename);
        else if (suffix == "txt")
            readSnapFile(filename);
        else if (suffix == "lvid")
            readLVIDFile(filename);
        else {
            std::cout << "Cannot read graph file based on its suffix ..." << std::endl;
            assert(false);
        }
    }
    else if (gt == GraphType::QUERY)
    {
        std::vector<std::vector<uintV>> conn = selectPresetPatterns(queryType);
        edge_count_ *= 2;
        row_ptrs_ = new uintE[vertex_count_ + 1];
        cols_ = new uintV[edge_count_];
        std::vector<uintE> offsets(vertex_count_ + 1, 0);

        for (size_t i = 0; i < vertex_count_; ++i)
        {
            offsets[i] = conn[i].size();
        }
        uintE prefix = 0;
        for (size_t i = 0; i < vertex_count_ + 1; ++i) {
            row_ptrs_[i] = prefix;
            prefix += offsets[i];
            offsets[i] = row_ptrs_[i];
        }
        for (size_t i = 0; i < vertex_count_; ++i)
        {
            for (const auto &j: conn[i])
            {
                cols_[offsets[j]++] = i;
            }
        }
        for (uintV u = 0; u < vertex_count_; ++u) {
            std::sort(cols_ + row_ptrs_[u], cols_ + row_ptrs_[u + 1]);
        }
        ReMapVertexId();
    }
    else
        assert(false);
}

void Graph::GetMaxDegree()
{
    ull max_deg = 0;
    for (size_t i = 0; i < vertex_count_; ++i)
    {
        max_deg = std::max(max_deg, row_ptrs_[i + 1] - row_ptrs_[i]);
    }
    std::cout << "max degree=" << max_deg << std::endl;
}

void Graph::ReMapVertexId()
{
    // 1. find the root vertex with largest degree
    size_t max_degree = 0;
    uintV root = 0;
    for (uintV i = 0; i < vertex_count_; i++) {
        if (row_ptrs_[i + 1] - row_ptrs_[i] > max_degree) {
            max_degree = row_ptrs_[i + 1] - row_ptrs_[i];
            root = i;
        }
    }
    // 2. bfs from the root vertex, make sure connected
    //    order: higher degree, more connections to the visited vertices
    std::queue<uintV> queue;
    std::vector<bool> visited(vertex_count_, false);
    queue.push(root);
    visited[root] = true;
    uintV new_vid = 0;
    std::vector<uintV> old_to_new(vertex_count_);
    std::vector<uintV> new_to_old(vertex_count_);
    uintE *new_row_ptrs_ = new uintE[vertex_count_ + 1];
    uintV *new_cols_ = new uintV[edge_count_];

    while (!queue.empty()) {
        size_t size = queue.size();
        std::vector<uintV> same_level_vertices;
        for (size_t i = 0; i < size; i++) {
            uintV front = queue.front();
            same_level_vertices.push_back(front);
            queue.pop();
            for (size_t j = row_ptrs_[front]; j < row_ptrs_[front + 1]; ++j)
            {  
                uintV ne = cols_[j];
                if (!visited[ne]) {
                    visited[ne] = true;
                    queue.push(ne);
                }
            }
        }
        std::vector<std::tuple<size_t, size_t, uintV>> weights;  // degree, connections, vid
        for (size_t i = 0; i < size; i++) {
            uintV v = same_level_vertices[i];
            size_t connections = 0;
            for (size_t j = row_ptrs_[v]; j < row_ptrs_[v + 1]; ++j)
            {
                uintV ne = cols_[j];
                if (visited[ne])
                    connections++;
            }
            weights.emplace_back(row_ptrs_[v + 1] - row_ptrs_[v], connections, v);
        }
        std::sort(weights.begin(), weights.end(), [](const auto& a, const auto& b) {
            if (std::get<0>(a) != std::get<0>(b))
                return std::get<0>(a) > std::get<0>(b);
            else if (std::get<1>(a) != std::get<1>(b))
                return std::get<1>(a) > std::get<1>(b);
            else if (std::get<2>(a) != std::get<2>(b))
                return std::get<2>(a) < std::get<2>(b);
            return false;
        });
        for (const auto& w : weights) {
            old_to_new[std::get<2>(w)] = new_vid;
            new_to_old[new_vid] = std::get<2>(w);
            new_vid++;
        }
    }
    auto offsets = new uintE[vertex_count_ + 1];
    memset(offsets, 0, sizeof(uintE) * (vertex_count_ + 1));

    for (size_t i = 0; i < vertex_count_; ++i)
    {
        offsets[i] = row_ptrs_[new_to_old[i] + 1] - row_ptrs_[new_to_old[i]];
    }
    uintE prefix = 0;
    for (size_t i = 0; i < vertex_count_ + 1; ++i) {
        new_row_ptrs_[i] = prefix;
        prefix += offsets[i];
        offsets[i] = new_row_ptrs_[i];
    }
    for (size_t i = 0; i < vertex_count_; ++i)
    {
        for (size_t j = row_ptrs_[i]; j < row_ptrs_[i + 1] ; ++j)
        {
            uintV ne = cols_[j];
            new_cols_[offsets[old_to_new[i]]++] = old_to_new[ne];
        }
    }
    for (uintV u = 0; u < vertex_count_; ++u) {
        std::sort(new_cols_ + new_row_ptrs_[u], new_cols_ + new_row_ptrs_[u + 1]);
    }
    delete[] offsets;
    delete[] row_ptrs_;
    delete[] cols_;
    row_ptrs_ = new_row_ptrs_;
    cols_ = new_cols_;
}

void Graph::readSnapFile(const std::string &filename)
{
    Timer timer;
    timer.StartTimer();

    vertex_count_ = 0;
    edge_count_ = 0;
    row_ptrs_ = NULL;
    cols_ = NULL;

    uintV min_vertex_id = std::numeric_limits<uintV>::max();
    uintV max_vertex_id = std::numeric_limits<uintV>::min();
    std::ifstream file(filename.c_str(), std::fstream::in);
    std::string line;
    uintV vids[2];
    while (getline(file, line)) {
        if (line.length() == 0 || !std::isdigit(line[0]))
            continue;
        std::istringstream iss(line);
        for (int i = 0; i < 2; ++i) {
            iss >> vids[i];
            min_vertex_id = std::min(min_vertex_id, vids[i]);
            max_vertex_id = std::max(max_vertex_id, vids[i]);
        }
        edge_count_++;
    }
    file.close();

    vertex_count_ = max_vertex_id - min_vertex_id + 1;
    edge_count_ *= 2;
    std::cout << "vertex_count=" << vertex_count_ << ", edge_count=" << edge_count_ << std::endl;

    row_ptrs_ = new uintE[vertex_count_ + 1];
    cols_ = new uintV[edge_count_];
    auto offsets = new uintE[vertex_count_ + 1];
    memset(offsets, 0, sizeof(uintE) * (vertex_count_ + 1));
    
    {
        std::ifstream file(filename.c_str(), std::fstream::in);
        std::string line;
        uintV vids[2];
        while (getline(file, line)) {
            if (line.length() == 0 || !std::isdigit(line[0]))
                continue;
            std::istringstream iss(line);
            for (int i = 0; i < 2; ++i)
            {
                iss >> vids[i];
                vids[i] -= min_vertex_id;
            }
            offsets[vids[0]]++;
            offsets[vids[1]]++;
        }
        file.close();
    }
    
    uintE prefix = 0;
    for (size_t i = 0; i < vertex_count_ + 1; ++i) {
        row_ptrs_[i] = prefix;
        prefix += offsets[i];
        offsets[i] = row_ptrs_[i];
    }

    {
        std::ifstream file(filename.c_str(), std::fstream::in);
        std::string line;
        uintV vids[2];
        while (getline(file, line)) {
            if (line.length() == 0 || !std::isdigit(line[0]))
                continue;
            std::istringstream iss(line);
            for (int i = 0; i < 2; ++i)
            {
                iss >> vids[i];
                vids[i] -= min_vertex_id;
            }
            cols_[offsets[vids[0]]++] = vids[1];
            cols_[offsets[vids[1]]++] = vids[0];
        }
        file.close();
    }
    delete[] offsets;
    offsets = NULL;

    for (uintV u = 0; u < vertex_count_; ++u) {
        std::sort(cols_ + row_ptrs_[u], cols_ + row_ptrs_[u + 1]);
    }

    timer.EndTimer();
    timer.PrintElapsedMicroSeconds("reading CSR Snap file");
}

void Graph::readBinFile(const std::string &filename)
{
    vertex_count_ = 0;
    edge_count_ = 0;
    row_ptrs_ = NULL;
    cols_ = NULL;

    Timer timer;
    timer.StartTimer();
    std::cout << "start reading CSR bin file...." << std::endl;
    FILE* file_in = fopen(filename.c_str(), "rb");
    assert(file_in != NULL);
    fseek(file_in, 0, SEEK_SET);
    size_t res = 0;
    size_t uintV_size = 0, uintE_size = 0;
    res += fread(&uintV_size, sizeof(size_t), 1, file_in);
    res += fread(&uintE_size, sizeof(size_t), 1, file_in);
    res += fread(&vertex_count_, sizeof(size_t), 1, file_in);
    res += fread(&edge_count_, sizeof(size_t), 1, file_in);
    assert(uintV_size == sizeof(uintV));
    assert(uintE_size == sizeof(uintE));
    std::cout << "vertex_count=" << vertex_count_ << ", edge_count=" << edge_count_ << std::endl;

    row_ptrs_ = new uintE[vertex_count_ + 1];
    cols_ = new uintV[edge_count_];
    res += fread(row_ptrs_, sizeof(uintE), vertex_count_ + 1, file_in);
    res += fread(cols_, sizeof(uintV), edge_count_, file_in);

    assert(res == 4 + (vertex_count_ + 1) + edge_count_);

    GetMaxDegree();

    fgetc(file_in);
    assert(feof(file_in));
    fclose(file_in);

    timer.EndTimer();
    timer.PrintElapsedMicroSeconds("reading CSR bin file");  
}

void Graph::readGraphFile(const std::string &filename)
{
    vertex_count_ = 0;
    edge_count_ = 0;
    row_ptrs_ = NULL;
    cols_ = NULL;

    std::ifstream file_in(filename);
    if (!file_in.is_open()) 
    {
        std::cout << "Unable to read the graph " << filename << std::endl;
    }

    char type;
    file_in >> type >> vertex_count_ >> edge_count_;
    edge_count_ *= 2;
    std::cout << "vertex_count=" << vertex_count_ << ", edge_count=" << edge_count_ << std::endl;

    row_ptrs_ = new uintE[vertex_count_ + 1];
    cols_ = new uintV[edge_count_];
    row_ptrs_[0] = 0;

    std::vector<uintV> neighbors_offset(vertex_count_, 0);
    
    while (file_in >> type) {
        if (type == 'v') { // Read vertex.
            ui id;
            ui label;
            ui degree;
            file_in >> id >> label >> degree;
            row_ptrs_[id + 1] = row_ptrs_[id] + degree;
        }
        else if (type == 'e') { // Read edge.
            uintV begin;
            uintV end;
            file_in >> begin >> end;

            uintV offset = row_ptrs_[begin] + neighbors_offset[begin];
            cols_[offset] = end;
            offset = row_ptrs_[end] + neighbors_offset[end];
            cols_[offset] = begin;
            neighbors_offset[begin] += 1;
            neighbors_offset[end] += 1;
        }
    }

    file_in.close();

    for (ui i = 0; i < vertex_count_; ++i) {
        std::sort(cols_ + row_ptrs_[i], cols_ + row_ptrs_[i + 1]);
    }
}

void Graph::readLVIDFile(const std::string &filename)
{
    vertex_count_ = 0;
    edge_count_ = 0;
    row_ptrs_ = NULL;
    cols_ = NULL;

    std::cout << "start build csr..." << std::endl;
    // const char* kDelimiters = " ,;\t";
    const char* kDelimiters = "0123456789";
    std::unordered_map<std::string, uintV> ids;
    std::vector<uintV> edge_pairs;
    {
        std::ifstream file(filename.c_str(), std::fstream::in);
        std::string line;
        while (getline(file, line)) {
            if (line.length() == 0 || !std::isdigit(line[0]))
                continue;

            std::vector<std::string> num_strs;
            size_t cur_pos = 0;
            while (cur_pos < line.length()) {
                cur_pos = line.find_first_of(kDelimiters, cur_pos, strlen(kDelimiters));
                if (cur_pos < line.length()) {
                    size_t next_pos = line.find_first_not_of(kDelimiters, cur_pos, strlen(kDelimiters));
                    num_strs.push_back(line.substr(cur_pos, next_pos - cur_pos));
                    assert(next_pos > cur_pos);
                    cur_pos = next_pos;
                }
            }

            for (auto& str : num_strs) {
                assert(str.length());
                for (auto ch : str) {
                    assert(std::isdigit(ch));
                }
            }

            for (auto& str : num_strs) {
                if (ids.find(str) == ids.end()) {
                    ids.insert(std::make_pair(str, vertex_count_++));
                }
                edge_pairs.push_back(ids[str]);
            }
        }
        file.close();
    }
    ids.clear();

    std::cout << "edge pairs size=" << edge_pairs.size() << std::endl;
    assert(edge_pairs.size() % 2 == 0);
    edge_count_ = edge_pairs.size(); // / 2;

    std::vector<uintE> offsets(vertex_count_ + 1, 0);
    for (size_t i = 0; i < edge_pairs.size(); i += 2) {
        offsets[edge_pairs[i]]++;
        offsets[edge_pairs[i + 1]]++;
    }

    row_ptrs_ = new uintE[vertex_count_ + 1];
    cols_ = new uintV[edge_count_];

    uintE prefix = 0;
    for (uintV i = 0; i <= vertex_count_; ++i) {
        row_ptrs_[i] = prefix;
        prefix += offsets[i];
        offsets[i] = row_ptrs_[i];
    }

    for (size_t i = 0; i < edge_pairs.size(); i += 2) 
    {
        cols_[offsets[edge_pairs[i]]++] = edge_pairs[i + 1];
        cols_[offsets[edge_pairs[i + 1]]++] = edge_pairs[i];
    }

    offsets.clear();
    edge_pairs.clear();

#pragma omp parallel for schedule(dynamic)
    for (uintV u = 0; u < vertex_count_; ++u) 
    {
        std::sort(cols_ + row_ptrs_[u], cols_ + row_ptrs_[u + 1]);
    }
    std::cout << "finish building CSR" << std::endl;
}

void Graph::writeBinFile(const std::string &filename)
{
    std::string prefix = filename.substr(0, filename.rfind("."));

    Timer timer;
    timer.StartTimer();
    std::cout << "start write CSR bin file...." << std::endl;
    std::string output_filename = prefix + ".bin";
    FILE* file_out = fopen(output_filename.c_str(), "wb");
    assert(file_out != NULL);
    size_t res = 0;
    size_t uintV_size = sizeof(uintV), uintE_size = sizeof(uintE);
    res += fwrite(&uintV_size, sizeof(size_t), 1, file_out);
    res += fwrite(&uintE_size, sizeof(size_t), 1, file_out);
    res += fwrite(&vertex_count_, sizeof(size_t), 1, file_out);
    res += fwrite(&edge_count_, sizeof(size_t), 1, file_out);
    res += fwrite(row_ptrs_, sizeof(uintE), vertex_count_ + 1, file_out);
    res += fwrite(cols_, sizeof(uintV), edge_count_, file_out);

    assert(res == 4 + (vertex_count_ + 1) + edge_count_);
    fclose(file_out);
    timer.EndTimer();
    timer.PrintElapsedMicroSeconds("writing CSR bin file");
}

bliss::Graph* Graph::GetBlissGraph()
{
    bliss::Graph* bg = new bliss::Graph(vertex_count_);
    for (size_t i = 0; i < vertex_count_; i++)
    {
        for (size_t j = row_ptrs_[i]; j < row_ptrs_[i + 1] ; j++)
            bg->add_edge(i, cols_[j]);
    }
    return bg;
}

std::vector<std::vector<uintV>> Graph::GetAutomorphisms(bliss::Graph* bg)
{
    std::vector<std::vector<uintV>> result;
    bliss::Stats stats;
    bg->find_automorphisms(
        stats,
        [](void* param, const ui size, const ui* aut) {
            std::vector<uintV> result_aut;
            for (ui i = 0; i < size; i++)
                result_aut.push_back(aut[i]);
            ((std::vector<std::vector<uintV>>*)param)->push_back(result_aut);
        },
        &result);

    uint32_t counter = 0;
    uint32_t lastSize = 0;
    while (result.size() != lastSize) 
    {
        lastSize = result.size();
        CompleteAutomorphisms(result);
        counter++;
        if (counter > 100)
            break;
    }

    return result;
}

std::string OrderToString(const std::vector<uint32_t>& p) {
    std::string res;
    for (auto v : p)
        res += std::to_string(v);
    return res;
}

void Graph::CompleteAutomorphisms(std::vector<std::vector<uint32_t>>& perm_group) 
{
    // multiplying std::vector<uint32_t>s is just function composition: (p1*p2)[i] = p1[p2[i]]
    std::vector<std::vector<uint32_t>> products;
    // for filtering duplicates
    std::unordered_set<std::string> dups;
    for (auto f : perm_group)
        dups.insert(OrderToString(f));

    for (auto k = perm_group.begin(); k != perm_group.end(); k++) 
    {
        for (auto l = perm_group.begin(); l != perm_group.end(); l++) 
        {
            std::vector<uint32_t> p1 = *k;
            std::vector<uint32_t> p2 = *l;

            std::vector<uint32_t> product;
            product.resize(p1.size());
            for (unsigned i = 0; i < product.size(); i++)
                product[i] = p1[p2[i]];

            // don't count duplicates
            if (dups.count(OrderToString(product)) == 0) 
            {
                dups.insert(OrderToString(product));
                products.push_back(product);
            }
        }
    }

    for (auto p : products)
        perm_group.push_back(p);
}

std::map<uintV, std::set<uintV>> Graph::GetAEquivalenceClasses(const std::vector<std::vector<uintV>>& aut) 
{
    std::map<uintV, std::set<uintV>> eclasses;
    for (size_t i = 0; i < vertex_count_; i++) 
    {
        std::set<uintV> eclass;
        for (auto&& perm : aut)
            eclass.insert(perm[i]);
        uintV rep = *std::min_element(eclass.cbegin(), eclass.cend());
        eclasses[rep].insert(eclass.cbegin(), eclass.cend());
    }
    return eclasses;
  }

std::vector<std::pair<uintV, uintV>> Graph::GetConditions(bliss::Graph* bg)
{
    std::vector<std::vector<uintV>> aut = GetAutomorphisms(bg);
    std::map<uintV, std::set<uintV>> eclasses = GetAEquivalenceClasses(aut);

    std::vector<std::pair<uintV, uintV>> result;
    auto eclass_it = std::find_if(eclasses.cbegin(), eclasses.cend(), [](auto&& e) { return e.second.size() > 1; });
    while (eclass_it != eclasses.cend() && eclass_it->second.size() > 1) 
    {
        const auto& eclass = eclass_it->second;
        uintV n0 = *eclass.cbegin();

        for (auto&& perm : aut)
        {
            uintV min = *std::min_element(std::next(eclass.cbegin()), eclass.cend(), [perm](uint32_t n, uint32_t m) { return perm[n] < perm[m]; });
            result.emplace_back(n0, min);
        }
        aut.erase(std::remove_if(aut.begin(), aut.end(), [n0](auto&& perm) { return perm[n0] != n0; }), aut.end());

        eclasses = GetAEquivalenceClasses(aut);
        eclass_it = std::find_if(eclasses.cbegin(), eclasses.cend(), [](auto&& e) { return e.second.size() > 1; });
    }

    // remove duplicate conditions
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());

    return result;
}


void Graph::SetConditions(const std::vector<std::pair<uintV, uintV>>& conditions) 
{
    order_.resize(vertex_count_);
    for (int i = 0; i < conditions.size(); i++) 
    {
        uintV first = conditions[i].first;
        uintV second = conditions[i].second;
        order_[first].push_back(std::make_pair(LESS_THAN, second));
        order_[second].push_back(std::make_pair(LARGER_THAN, first));
    }
}


std::string GetCondOperatorString(const CondOperator& op) {
    std::string ret = "";
    switch (op) {
        case LESS_THAN:
            ret = "LESS_THAN";
            break;
        case LARGER_THAN:
            ret = "LAGER_THAN";
            break;
        case NON_EQUAL:
            ret = "NON_EQUAL";
            break;
        default:
            break;
    }
    return ret;
}

void Graph::Preprocess() {
    // remove dangling nodes
    // self loops
    // parallel edges
    Timer timer;
    timer.StartTimer();
    std::cout << "start preprocess..." << std::endl;
    size_t vertex_count = GetVertexCount();
    size_t edge_count = GetEdgeCount();
    auto row_ptrs = GetRowPtrs();
    auto cols = GetCols();

    auto vertex_ecnt = new uintE[vertex_count + 1];
    memset(vertex_ecnt, 0, sizeof(uintE) * (vertex_count + 1));
    for (uintV u = 0; u < vertex_count; ++u) 
    {
        for (auto j = row_ptrs[u]; j < row_ptrs[u + 1]; ++j) 
        {
            auto v = cols[j];
            bool parallel_edge = (j > row_ptrs[u] && v == cols[j - 1]);
            bool self_loop = u == v;
            if (!parallel_edge && !self_loop) 
            {
                vertex_ecnt[u]++;
            }
        }
    }
    auto nrow_ptrs = new uintE[vertex_count + 1];
    uintE nedge_count = 0;
    for (uintV u = 0; u < vertex_count; ++u) 
    {
        nrow_ptrs[u] = nedge_count;
        nedge_count += vertex_ecnt[u];
    }
    nrow_ptrs[vertex_count] = nedge_count;
    delete[] vertex_ecnt;
    vertex_ecnt = NULL;

    auto ncols = new uintV[nedge_count];
    for (uintV u = 0; u < vertex_count; ++u) 
    {
        auto uoff = nrow_ptrs[u];
        for (uintE j = row_ptrs[u]; j < row_ptrs[u + 1]; ++j) 
        {
            auto v = cols[j];
            bool parallel_edge = j > row_ptrs[u] && v == cols[j - 1];
            bool self_loop = u == v;
            if (!parallel_edge && !self_loop) 
            {
                ncols[uoff++] = v;
            }
        }
    }
    edge_count = nedge_count;
    std::swap(row_ptrs, nrow_ptrs);
    std::swap(cols, ncols);
    delete[] nrow_ptrs;
    nrow_ptrs = NULL;
    delete[] ncols;
    ncols = NULL;

    auto new_vertex_ids = new uintV[vertex_count];
    uintV max_vertex_id = 0;
    for (uintV u = 0; u < vertex_count; ++u) 
    {
        if (row_ptrs[u] == row_ptrs[u + 1]) {
            new_vertex_ids[u] = vertex_count;
        } 
        else 
        {
            new_vertex_ids[u] = max_vertex_id++;
            row_ptrs[new_vertex_ids[u]] = row_ptrs[u];
        }
    }
    for (uintE j = 0; j < edge_count; ++j) 
    {
        cols[j] = new_vertex_ids[cols[j]];
    }
    delete[] new_vertex_ids;
    new_vertex_ids = NULL;
    vertex_count = max_vertex_id;
    row_ptrs[vertex_count] = edge_count;

    timer.EndTimer();
    std::cout << "finish preprocess, time=" << timer.GetElapsedMicroSeconds() / 1000.0 << "ms"
              << ", now vertex_count=" << vertex_count << ",edge_count=" << edge_count << std::endl;

    SetVertexCount(vertex_count);
    SetEdgeCount(edge_count);
    SetRowPtrs(row_ptrs);
    SetCols(cols);
}

std::vector<std::vector<uintV>> Graph::selectPresetPatterns(PresetPatternType patternType)
{
    std::vector<std::vector<uintV>> conn_;
    switch (patternType) 
    {
    case P0:
        vertex_count_ = 3;
        edge_count_ = 3;

        conn_.push_back({1, 2});  // 0
        conn_.push_back({0, 2});  // 1
        conn_.push_back({0, 1});  // 2
        break;
    case P1:
        // square
        vertex_count_ = 4;
        edge_count_ = 4;

        conn_.push_back({1, 3});  // 0
        conn_.push_back({0, 2});  // 1
        conn_.push_back({1, 3});  // 2
        conn_.push_back({0, 2});  // 3
        break;
    case P2:
        // chrodal square
        vertex_count_ = 4;
        edge_count_ = 5;

        conn_.push_back({1, 2, 3});  // 0
        conn_.push_back({0, 2});     // 1
        conn_.push_back({0, 1, 3});  // 2
        conn_.push_back({0, 2});     // 3
        break;
    case P3:
        // 2 tails triangle
        vertex_count_ = 5;
        edge_count_ = 5;

        conn_.push_back({1, 2});     // 0
        conn_.push_back({0, 2});     // 1
        conn_.push_back({0, 1, 3});  // 2
        conn_.push_back({2, 4});     // 3
        conn_.push_back({3});        // 4
        break;
      case P4:
        // house
        vertex_count_ = 5;
        edge_count_ = 6;

        conn_.push_back({1, 2});     // 0
        conn_.push_back({0, 2, 3});  // 1
        conn_.push_back({0, 1, 4});  // 2
        conn_.push_back({1, 4});     // 3
        conn_.push_back({2, 3});     // 4
        break;
      case P5:
        // chrodal house
        vertex_count_ = 5;
        edge_count_ = 8;

        conn_.push_back({1, 2, 3, 4});  // 0
        conn_.push_back({0, 2, 3});     // 1
        conn_.push_back({0, 1, 3});     // 2
        conn_.push_back({0, 1, 2, 4});  // 3
        conn_.push_back({0, 3});        // 4
        break;
      case P6:
        // chrodal roof
        vertex_count_ = 5;
        edge_count_ = 7;

        conn_.push_back({1, 2, 3});  // 0
        conn_.push_back({0, 2, 3});  // 1
        conn_.push_back({0, 1, 4});  // 2
        conn_.push_back({0, 1, 4});  // 3
        conn_.push_back({2, 3});     // 4
        break;
      case P7:
        // three triangles
        vertex_count_ = 5;
        edge_count_ = 7;

        conn_.resize(vertex_count_);
        for (uintV i = 1; i <= 4; ++i) {
          conn_[0].push_back(i);
          conn_[i].push_back(0);
        }
        conn_[1].push_back(3);
        conn_[3].push_back(1);
        conn_[1].push_back(2);
        conn_[2].push_back(1);
        conn_[2].push_back(4);
        conn_[4].push_back(2);
        break;
      case P8:
        // solar square
        vertex_count_ = 5;
        edge_count_ = 8;

        conn_.push_back({1, 2, 3, 4});  // 0
        conn_.push_back({0, 2, 4});     // 1
        conn_.push_back({0, 1, 3});     // 2
        conn_.push_back({0, 2, 4});     // 3
        conn_.push_back({0, 1, 3});     // 4
        break;
      case P9:
        // near 5 clique
        vertex_count_ = 5;
        edge_count_ = 9;

        conn_.push_back({1, 2, 3, 4});  // 0
        conn_.push_back({0, 2, 3, 4});  // 1
        conn_.push_back({0, 1, 3});     // 2
        conn_.push_back({0, 1, 2, 4});  // 3
        conn_.push_back({0, 1, 3});     // 4
        break;
      case P10:
        // four triangles
        vertex_count_ = 6;
        edge_count_ = 9;

        conn_.push_back({1, 2, 3, 4, 5});  // 0
        conn_.push_back({0, 2});           // 1
        conn_.push_back({0, 1, 3});        // 2
        conn_.push_back({0, 2, 4});        // 3
        conn_.push_back({0, 3, 5});        // 4
        conn_.push_back({0, 4});           // 5

        break;
      case P11:
        // one in three triangles
        vertex_count_ = 6;
        edge_count_ = 9;

        conn_.push_back({1, 2, 3, 5});  // 0
        conn_.push_back({0, 2, 3, 4});  // 1
        conn_.push_back({0, 1, 4, 5});  // 2
        conn_.push_back({0, 1});        // 3
        conn_.push_back({1, 2});        // 4
        conn_.push_back({0, 2});        // 5

        break;
      case P12:
        // near 6 clique
        vertex_count_ = 6;
        edge_count_ = 11;

        conn_.push_back({1, 2});           // 0
        conn_.push_back({0, 2, 3, 4, 5});  // 1
        conn_.push_back({0, 1, 3, 4, 5});  // 2
        conn_.push_back({1, 2, 4});        // 3
        conn_.push_back({1, 2, 3, 5});     // 4
        conn_.push_back({1, 2, 4});        // 5

        break;
      case P13:
        // square on top
        vertex_count_ = 6;
        edge_count_ = 8;

        conn_.push_back({1, 2});        // 0
        conn_.push_back({0, 3});        // 1
        conn_.push_back({0, 3, 4, 5});  // 2
        conn_.push_back({1, 2, 4, 5});  // 3
        conn_.push_back({2, 3});        // 4
        conn_.push_back({2, 3});        // 5

        break;
      case P14:
        // near 7 clique
        vertex_count_ = 7;
        edge_count_ = 15;

        conn_.push_back({1, 2, 3, 4, 5});     // 0
        conn_.push_back({0, 2, 3, 5});        // 1
        conn_.push_back({0, 1, 3, 5});        // 2
        conn_.push_back({0, 1, 2, 4, 5, 6});  // 3
        conn_.push_back({0, 3, 5});           // 4
        conn_.push_back({0, 1, 2, 3, 4, 6});  // 5
        conn_.push_back({3, 5});              // 6
        break;
      case P15:
        // 5 clique on top
        vertex_count_ = 7;
        edge_count_ = 14;

        conn_.push_back({1, 2, 3, 4});        // 0
        conn_.push_back({0, 2, 3, 4});        // 1
        conn_.push_back({0, 1, 3, 4});        // 2
        conn_.push_back({0, 1, 2, 4, 5, 6});  // 3
        conn_.push_back({0, 1, 2, 3, 5, 6});  // 4
        conn_.push_back({3, 4});              // 5
        conn_.push_back({3, 4});              // 6
        break;
      case P16:
        // 5 circles
        vertex_count_ = 5;
        edge_count_ = 5;

        conn_.push_back({1, 2});  // 0
        conn_.push_back({0, 3});  // 1
        conn_.push_back({0, 4});  // 2
        conn_.push_back({1, 4});  // 3
        conn_.push_back({2, 3});  // 4

        break;
      case P17:
        // 6 circles
        vertex_count_ = 6;
        edge_count_ = 6;

        conn_.push_back({1, 2});  // 0
        conn_.push_back({0, 3});  // 1
        conn_.push_back({0, 4});  // 2
        conn_.push_back({1, 5});  // 3
        conn_.push_back({2, 5});  // 4
        conn_.push_back({3, 4});  // 5
        break;
      case P18:
        // hourglass
        vertex_count_ = 6;
        edge_count_ = 9;

        conn_.push_back({1, 2, 4});  // 0
        conn_.push_back({0, 2, 5});  // 1
        conn_.push_back({0, 1, 3});  // 2
        conn_.push_back({2, 4, 5});  // 3
        conn_.push_back({0, 3, 5});  // 4
        conn_.push_back({1, 3, 4});  // 5
        break;

      case P23:
        // 3 clique
        vertex_count_ = 3;
        edge_count_ = 3;

        conn_.push_back({1, 2});  // 0
        conn_.push_back({0, 2});  // 1
        conn_.push_back({0, 1});  // 2
        break;
      case P24:
        // 4 clique
        vertex_count_ = 4;
        edge_count_ = 6;

        conn_.push_back({1, 2, 3});  // 0
        conn_.push_back({0, 2, 3});  // 1
        conn_.push_back({0, 1, 3});  // 2
        conn_.push_back({0, 1, 2});  // 3
        break;
      case P25:
        // 5 clique
        vertex_count_ = 5;
        edge_count_ = 10;

        conn_.push_back({1, 2, 3, 4});  // 0
        conn_.push_back({0, 2, 3, 4});  // 1
        conn_.push_back({0, 1, 3, 4});  // 2
        conn_.push_back({0, 1, 2, 4});  // 3
        conn_.push_back({0, 1, 2, 3});  // 4
        break;
      case P26:
        // 6 clique
        vertex_count_ = 6;
        edge_count_ = 15;

        conn_.push_back({1, 2, 3, 4, 5});  // 0
        conn_.push_back({0, 2, 3, 4, 5});  // 1
        conn_.push_back({0, 1, 3, 4, 5});  // 2
        conn_.push_back({0, 1, 2, 4, 5});  // 3
        conn_.push_back({0, 1, 2, 3, 5});  // 4
        conn_.push_back({0, 1, 2, 3, 4});  // 5
        break;
      case P27:
        // 7 clique
        vertex_count_ = 7;
        edge_count_ = 21;

        conn_.push_back({1, 2, 3, 4, 5, 6});  // 0
        conn_.push_back({0, 2, 3, 4, 5, 6});  // 1
        conn_.push_back({0, 1, 3, 4, 5, 6});  // 2
        conn_.push_back({0, 1, 2, 4, 5, 6});  // 3
        conn_.push_back({0, 1, 2, 3, 5, 6});  // 4
        conn_.push_back({0, 1, 2, 3, 4, 6});  // 5
        conn_.push_back({0, 1, 2, 3, 4, 5});  // 6
        break;
      default:
        assert(false);
        break;
    }
    return conn_;
}

#endif // COMMON_GRPAH_H