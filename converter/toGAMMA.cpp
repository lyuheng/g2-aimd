#include "common/graph.h"

int main(int argc, char *argv[])
{   
    std::string filename = std::string(argv[1]);
    Graph graph(filename);

    std::string prefix = filename.substr(0, filename.rfind("."));

    {
        std::string output_offset = prefix + ".col";
        FILE* file_out = fopen(output_offset.c_str(), "wb");
        assert(file_out != NULL);
        size_t res = 0;
        ui vertex_count = graph.GetVertexCount();
        res += fwrite(&vertex_count, sizeof(ui), 1, file_out);
        for (size_t i = 0; i < graph.GetVertexCount() + 1; ++i)
        {
            res += fwrite(&(graph.GetRowPtrs()[i]), sizeof(uintE), 1, file_out);
        }
        fclose(file_out);
    }

    {
        std::string output_vertices = prefix + ".dst";
        FILE* file_out = fopen(output_vertices.c_str(), "wb");

        assert(file_out != NULL);
        size_t res = 0;
        size_t edge_count = graph.GetEdgeCount();
        res += fwrite(&edge_count, sizeof(size_t), 1, file_out);
        for (size_t i = 0; i < graph.GetEdgeCount(); ++i)
        {
            res += fwrite(&(graph.GetCols()[i]), sizeof(ui), 1, file_out);
        }
        fclose(file_out);
    }

    {
        std::string output_vertices = prefix + ".vlabel";
        FILE* file_out = fopen(output_vertices.c_str(), "wb");

        assert(file_out != NULL);
        size_t res = 0;
        size_t edge_count = graph.GetEdgeCount();
        size_t vertex_count = graph.GetVertexCount();
        res += fwrite(&vertex_count, sizeof(uint32_t), 1, file_out);
        uint8_t label = 1;
        for (size_t i = 0; i < vertex_count; ++i)
        {
            res += fwrite(&label, sizeof(uint8_t), 1, file_out);
        }
        fclose(file_out);
    }
}