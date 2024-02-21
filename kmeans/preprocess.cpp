#include "common/graph.h"
#include "common/command_line.h"

int main(int argc, char *argv[])
{
        Timer timer;
    timer.StartTimer();
    CommandLine cmd(argc, argv);
    std::string filename = cmd.GetOptionValue("-f", "./data/com-dblp.ungraph.txt");
    Graph graph(filename);
    
    graph.Preprocess();
    graph.writeBinFile(filename);
    timer.EndTimer();
    timer.PrintElapsedMicroSeconds("preprocessing completed: ");
}
