#ifndef SYSTEM_WORKER_H
#define SYSTEM_WORKER_H

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <string>

#include "common/command_line.h"
#include "view/view_bin_manager.h"
#include "view/view_bin_buffer.h"
#include "rigtorp/MPMCQueue.h"
#include "system/pipeline_executor.h"
#include "system/appbase.h"
#include "system/subgraph_container.h"
#include "system/buffer.h"
#include "system/plan.h"

//1158240 = 193040*6 (now 193040)
template<typename Application>
class Worker
{
public:
    void run(int argc, char *argv[], Application app)
    {
        // deviceQuery();
        CommandLine cmd(argc, argv);
        std::string data_filename = cmd.GetOptionValue("-dg", "./data/com-friendster.ungraph.txt.bin");
        int query_type = cmd.GetOptionIntValue("-q", 1000);
        double mem = cmd.GetOptionDoubleValue("-m", 10);
        int thread_num = cmd.GetOptionIntValue("-t", 1);
        int queue_size = cmd.GetOptionIntValue("-qs", 1);
        int hop = cmd.GetOptionIntValue("-h", -1); // -1 means worker needs to analyse hop by itself.
        int producers_num = cmd.GetOptionIntValue("-pn", 1);
        int consumers_num = cmd.GetOptionIntValue("-cn", 1);
        int do_reorder = cmd.GetOptionIntValue("-dr", 0);
        int do_split = cmd.GetOptionIntValue("-ds", 0);
        int do_split_times = cmd.GetOptionIntValue("-dst", 1);
        int sort_sources = cmd.GetOptionIntValue("-ss", 0);

        int gpu_count = 0;
        cudaGetDeviceCount(&gpu_count);
        assert(consumers_num <= gpu_count);
        std::cout << "GPU count: " << gpu_count << std::endl;

        assert(queue_size >= consumers_num);
        std::cout << "m: " << mem << " t: " << thread_num << " qs: " << queue_size << " pn: " << producers_num << " cn: " << consumers_num << std::endl;

        Graph *graph = new Graph(data_filename);
        int root_degree = 0;
        std::vector<StoreStrategy> strategy;



        if (query_type != 1000)
        {
            assert(hop == -1);
            Graph query_G("", (PresetPatternType)query_type, GraphType::QUERY);
            query_G.SetConditions(query_G.GetConditions(query_G.GetBlissGraph()));

            auto& order = query_G.order_;
            std::cout << "conditions: " << std::endl;
            for (ui i = 0; i < order.size(); i++) 
            {
                std::cout << i << ": ";
                for (ui j = 0; j < order[i].size(); j++)
                    std::cout << GetCondOperatorString(order[i][j].first) << "(" << order[i][j].second << "), ";
                std::cout << std::endl;
            }

            Plan plan;
            plan.graph = std::move(query_G);

            // TODO: FIXME: compute hop
            // hop = 2; // test pattern specific for a triangle pattern

            plan.FindRoot();
            plan.GenerateSearchSequence();
            plan.GenerateBackwardNeighbor();
            plan.GeneratePreAfterBackwardNeighbor();
            plan.GenerateUsefulOrder();
            plan.GenerateStoreStrategy();
            hop = plan.GetHop();
            root_degree = plan.root_degree_;
            strategy = plan.strategy;

            std::cout << "Hop = " << hop << std::endl;
            std::cout << "Root degree = " << root_degree << std::endl;
            std::cout << "Match Order: ";
            for (size_t i = 0; i < plan.vertex_count_; ++i)
            {
                std::cout << plan.seq_[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "Moving Level: ";
            for (size_t i = 0; i < plan.vertex_count_; ++i)
            {
                std::cout << plan.moving_lvl[i] << " ";
            }
            std::cout << std::endl;

            for (size_t i = 0; i < plan.vertex_count_; ++i)
            {
                std::cout << i << ": ";
                for (size_t j = 0; j < plan.backNeighborCountHost[plan.reverse_seq_[i]]; ++j)
                {
                    std::cout << plan.backNeighborsHost[plan.reverse_seq_[i] * plan.vertex_count_ + j] << " ";
                }
                std::cout << std::endl;
            }

            for (size_t i = 0; i < plan.vertex_count_; ++i)
            {
                if (plan.share_intersection[i])
                {
                    std::cout << "shared loc = " << i << "       ";
                    std::cout << "pre BN: ";
                    for (size_t j = 0; j < plan.preBackNeighborCountHost[i]; ++j)
                    {
                        std::cout << plan.preBackNeighborsHost[i * plan.vertex_count_ + j] << " ";
                    }
                    std::cout << "after BN: ";
                    for (size_t j = 0; j < plan.afterBackNeighborCountHost[i]; ++j)
                    {
                        std::cout << plan.afterBackNeighborsHost[i * plan.vertex_count_ + j] << " ";
                    }
                    std::cout << std::endl;
                }
            }

            for (size_t i = 0; i < plan.vertex_count_; ++i)
            {
                if (plan.share_intersection[plan.seq_[i]])
                {
                    std::cout << "shared VertexID = " << i << "       ";
                    std::cout << "pre Cond: ";
                    for (size_t j = 0; j < plan.preCondNumHost[i]; ++j)
                    {
                        std::cout << plan.preCondOrderHost[2 * i * plan.vertex_count_ + j] 
                            << " " << plan.preCondOrderHost[2 * i * plan.vertex_count_ + j + 1] << " ";
                    }
                    std::cout << "after Cond: ";
                    for (size_t j = 0; j < plan.afterCondNumHost[i]; ++j)
                    {
                        std::cout << plan.afterCondOrderHost[2 * i * plan.vertex_count_ + j] 
                            << " " << plan.afterCondOrderHost[2 * i * plan.vertex_count_ + j + 1] << " ";
                    }
                    std::cout << std::endl;
                }
            }

            app.context.AddGMContext(plan.seq_, plan.reverse_seq_, plan.backNeighborCountHost,
                                    plan.backNeighborsHost, plan.parentHost, plan.vertex_count_,
                                    plan.condOrderHost, plan.condNumHost, plan.share_intersection,
                                    plan.preBackNeighborCountHost, plan.preBackNeighborsHost, plan.preCondOrderHost,
                                    plan.preCondNumHost, plan.afterBackNeighborCountHost, plan.afterBackNeighborsHost,
                                    plan.afterCondOrderHost, plan.afterCondNumHost, plan.strategy, plan.moving_lvl);
        }

        // Load view bins partition
        ViewBinManager *vbm = new ViewBinManager(graph, hop, thread_num);
        std::string partition_file = data_filename + "." + std::to_string(hop) + ".hop.vbmap";
        vbm->LoadViewBinPartition(partition_file);
        if (sort_sources)
            vbm->SortSources();
        if (do_reorder)
            vbm->Reorder();
        if (do_split)
            vbm->Split(consumers_num, do_split_times);
        size_t max_partitioned_sources_num = vbm->GetMaxPartitionedSourcesNum();
        size_t max_view_bin_size = mem * 1000 * 1000 * 1000; // already the total size, no need to multiply 4 bytes

        ViewBinBuffer view_bin_buffer(queue_size, graph->GetVertexCount(), max_view_bin_size);

        rigtorp::MPMCQueue<int> assigned_queue(queue_size);
        rigtorp::MPMCQueue<int> released_queue(queue_size);
        // fill up the queue first
        for (int i = 0; i < queue_size; i++)
            released_queue.push(i);

        std::vector<std::thread> threads;
        std::atomic<int> view_bin_pool_index{0};
        std::atomic<int> num_finished_producers{0};
        auto &view_bin_pool = vbm->GetViewBinPool();

        for(ui i = 0; i < view_bin_pool.size(); ++i)
            std::cout << i << " " << view_bin_pool[i]->GetId() << std::endl;


        // Multiple Producers
        for (int p = 0; p < producers_num; p++)
        {
            threads.push_back(std::thread([p, queue_size, root_degree, &num_finished_producers, &view_bin_pool_index, &view_bin_pool, &view_bin_buffer, &assigned_queue, &released_queue]() {
                for (int view_bin_id{}; (view_bin_id = view_bin_pool_index++) < view_bin_pool.size();) {
                    // if (view_bin_id < 9) continue;
                    int holder_id = -1;
                    released_queue.pop(holder_id);
                    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));                
                    std::cout << "producer " << p << " gets " << holder_id << ", view_bin_id = " << view_bin_id << std::endl;
                    view_bin_pool[view_bin_id]->Materialize(view_bin_buffer.GetViewBinHolder(holder_id), root_degree);
                    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));  
                    std::cout << "producer " << p << " produces " << view_bin_id << " in " << holder_id << std::endl;
                    assigned_queue.push(holder_id);
                }
                // signal to stop the consumers
                num_finished_producers++;
            }));
        }

        std::vector<std::vector<ull>> counts(consumers_num);

        for (int c = 0; c < consumers_num; c++)
        {
            threads.push_back(std::thread([c, producers_num, max_partitioned_sources_num, &graph, &num_finished_producers, &view_bin_buffer, &assigned_queue, &released_queue, &counts, app, max_view_bin_size, &strategy]() {
                PipelineExecutor<Application>* executor = new PipelineExecutor<Application>(c, graph, max_partitioned_sources_num, app, max_view_bin_size, strategy);
                while (true) {
                    int holder_id = -1;
                    if (num_finished_producers == producers_num) {  // stop condition
                        // std::this_thread::sleep_for(std::chrono::milliseconds(10));  // avoid busy waiting
                        if (!assigned_queue.try_pop(holder_id))  // try pop again and exit if no more items
                            break;
                    } else if (!assigned_queue.try_pop(holder_id)) {  // try pop if empty
                        // std::this_thread::sleep_for(std::chrono::milliseconds(10));  // avoid busy waiting
                        continue;
                    }

                    // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                    std::cout << "consumer " << c << " gets " << holder_id << std::endl;
                    executor->Transfer(view_bin_buffer.GetViewBinHolder(holder_id));
                    std::cout << "consumer " << c << " completes transfer " << holder_id << std::endl;
                    // can release after transfer
                    released_queue.push(holder_id);

                    Timer in_memory_timer;
                    in_memory_timer.StartTimer();

                    executor->Run();

                    in_memory_timer.EndTimer();
                    in_memory_timer.PrintElapsedMicroSeconds("In-memory Execution Time");


                    std::cout << "consumer " << c << " completes match " << holder_id << std::endl;
                    // executor->PrintTotalCounts();

                    size_t total_result = 0;
                    for (ui i = 0; i < N_WARPS; ++i)
                    {
                        total_result += executor->app.total_counts_[i];
                    }
                    std::cout << "Currently Total counts: " << total_result << std::endl;

                }
                // counts[c] = executor->GetTotalCounts();
                counts[c].assign(executor->app.total_counts_, executor->app.total_counts_ + N_WARPS);
                std::cout << "consumer " << c << " stops" << std::endl; 
            }));
        }

        Timer timer;
        timer.StartTimer();

        for (auto& t : threads)
            t.join();

        timer.EndTimer();
        timer.PrintElapsedMicroSeconds("total time");

        size_t total_result = 0;
        for (ui i = 0; i < consumers_num; ++i)
        {
            for (ui j = 0; j < N_WARPS; ++j)
                total_result += counts[i][j];
        }
        std::cout << "Total counts: " << total_result << std::endl;
    }
};

#endif