#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <stack>
#include <vector>

#include "common/command_line.h"
#include "common/meta.h"
#include "common/graph.h"
#include "system/worker.h"
#include "BK.h"

int main(int argc, char *argv[])
{
    BKExpandSequential app;
    Worker<BKExpandSequential> worker;

    worker.run(argc, argv, app);

    return 0;
}

