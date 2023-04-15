#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <stack>
#include <vector>

#include "common/command_line.h"
#include "common/meta.h"
#include "common/graph.h"
#include "system/worker.h"
#include "gmatch.h"

int main(int argc, char *argv[])
{
    GMatchApp app;
    Worker<GMatchApp> worker;
    worker.run(argc, argv, app);
    
    return 0;
}

