#include "common/graph.h"
#include <fstream>

int main(int argc, char *argv[])
{   
    std::string filename = std::string(argv[1]);
    Graph graph(filename);

    std::string prefix = filename.substr(0, filename.rfind("."));

    std::string output_file_name = prefix + ".pangolin.txt";

    std::ofstream outfile;
    outfile.open(output_file_name, std::ofstream::trunc);

    for (size_t i = 0; i < graph.GetVertexCount(); ++i)
    {
        for (size_t j = graph.GetRowPtrs()[i]; j < graph.GetRowPtrs()[i + 1]; ++j)
        {
            if (i < graph.GetCols()[j])
                outfile << i + 1 << " " << graph.GetCols()[j] + 1 << std::endl;
        }
    }
    outfile.close();
}