#include "system/appbase.h"
#include "system/buffer.h"
#include "common/meta.h"
#include "system/util.h"

template<class IndexType>
class BKBuffer : public BufferBase<IndexType>
{
public:
    Label *labels;

    static size_t sizeOf()
    {
        return BufferBase<IndexType>::sizeOf() + sizeof(Label);
    }
    void allocateMemory(size_t sz)
    {
        BufferBase<IndexType>::allocateMemory(sz);
        chkerr(cudaMalloc((void **)&labels, sz * sizeof(Label)));
    }

    /**
     * @brief This version is used to allocate memory on host. Call it only for HOST_BUFF_SZ
     *
     */
    void allocateMemory()
    {
        BufferBase<IndexType>::allocateMemory();
        chkerr(cudaMallocManaged((void **)&labels, HOST_BUFF_SZ * sizeof(Label)));
    }
};

class BKBase : public AppBase<BKBuffer>
{
    unsigned int *tempv;
    char *templ;

public:
    unsigned long long int *iterCliques;
    unsigned long long int *totalCliques;


    
    unsigned int firstRoundIterNumHost(){
        // do nothing
        return dp->V;
    }

    void allocateMemory()
    {

        chkerr(cudaMalloc((void **)&tempv, TEMPSIZE * N_WARPS * sizeof(unsigned int)));
        chkerr(cudaMalloc((void **)&templ, TEMPSIZE * N_WARPS * sizeof(char)));

        chkerr(cudaMallocManaged((void **)&iterCliques, sizeof(int)));
        chkerr(cudaMallocManaged((void **)&totalCliques, sizeof(int)));
        totalCliques[0] = 0;
        iterCliques[0] = 0;
    }

    void iterationFailed()
    {
        iterCliques[0] = 0;
    }
    void iterationSuccess()
    {
        totalCliques[0] += iterCliques[0];
        iterCliques[0] = 0;
        cout << " cliques " << totalCliques[0] << endl;
    }

    void completion()
    {
        cout << "Total No. of Cliques: " << totalCliques[0] << endl;
    }

    __device__ bool examineClique(SubgraphOffsets *so)
    {
        auto st = so->st;
        auto en = so->en;
        auto data = sg->rdBuff.labels;
        for (; st < en; st += 32)
        {
            unsigned int k = st + LANEID; // want to include all lanes.
            bool pred = k < en && (data[k] == P || data[k] == X);
            if (__ballot_sync(FULL, pred))
                return false;
        }
        return true;
    }

    __device__ bool crossed(SubgraphOffsets *so)
    {
        auto st = so->st;
        auto en = so->en;
        auto data = sg->rdBuff.labels;
        for (; st < en; st += 32)
        {
            unsigned int k = st + LANEID; // want to include all lanes.
            bool pred = k < en && (data[k] == P);
            if (__ballot_sync(FULL, pred))
                return false;
        }
        return true;
    }

    __device__ unsigned int writeToTemp(unsigned int v, char label, bool pred, unsigned int sglen)
    {
        unsigned int loc = scanIndex(pred) + sglen;
        // popc gives inclusive sum scan, subtract pred to make it exclusive
        // add sglen to find exact location in the temp
        assert(loc < TEMPSIZE);
        if (pred)
        {
            this->tempv[loc + GLWARPID * TEMPSIZE] = v;
            this->templ[loc + GLWARPID * TEMPSIZE] = label;
        }
        if (LANEID == 31) // last lane's loc+pred is number of items found in this scan
            sglen = loc + pred;
        sglen = __shfl_sync(FULL, sglen, 31);
        return sglen;
    }

    // this overloaded form doesn't write the labels.
    // this version is called when separating P in PivotSelection
    __device__ unsigned int writeToTemp(unsigned int v, bool pred, unsigned int sglen)
    {
        unsigned int loc = scanIndex(pred) + sglen;
        // popc gives inclusive sum scan, subtract pred to make it exclusive
        // add sglen to find exact location in the temp
        assert(loc < TEMPSIZE);
        if (pred)
            this->tempv[loc + GLWARPID * TEMPSIZE] = v;
        if (LANEID == 31) // last lane's loc+pred is number of items found in this scan
            sglen = loc + pred;
        sglen = __shfl_sync(FULL, sglen, 31);
        return sglen;
    }

    __device__ int getSubgraphTemp(SubgraphOffsets *so, unsigned int q)
    {
        // unsigned int laneid = LANEID;
        auto st = so->st;
        auto en = so->en;
        // printf("#%u:%u:%u*", s, st, en);
        unsigned int qst = dp->neighbors_offset[q];
        unsigned int qen = dp->neighbors_offset[q + 1];
        unsigned int v, sglen = 0;
        char label;
        bool pred;

        // ### Binary search, it'll also need a scan to get the locations.
        for (; st < en; st += 32)
        {
            unsigned int i = st + LANEID;
            pred = false; // if i>=en, then pred will be false...
            if (i < en)
            {
                v = sg->rdBuff.vertices[i];
                label = sg->rdBuff.labels[i];
                // no need to intersect R nodes
                pred = (label == R) || binarySearch(dp->neighbors, qst, qen, v);
            }
            sglen = writeToTemp(v, label, pred, sglen); // appply sum scan and store in temp...
            // sglen is passed by reference to this function, and it gets the length of subgraph
        }
        return sglen;
    }
    __device__ void generateSubgraphDoubleIntersect(SubgraphOffsets *so, unsigned int q)
    {
        auto st = so->st;
        auto en = so->en;
        // printf("#%u:%u:%u*", s, st, en);
        unsigned int qst = dp->neighbors_offset[q];
        unsigned int qen = dp->neighbors_offset[q + 1];
        unsigned int v, sglen = 0;
        char label;
        bool pred;

        // Perform intersection to find length of subgraph
        for (; st < en; st += 32)
        {
            unsigned int i = st + LANEID;
            pred = false; // if i>=en, then pred will be false...
            if (i < en)
            {
                v = sg->rdBuff.vertices[i];
                label = sg->rdBuff.labels[i];
                // no need to intersect R nodes
                pred = (label == R) || binarySearch(dp->neighbors, qst, qen, v);
            }
            sglen += __popc(__ballot_sync(FULL, pred)); // appply sum scan and store in temp...
            // sglen is passed by reference to this function, and it gets the length of subgraph
        }

        unsigned int vt = sg->append(sglen + 1);
        if (LANEID == 0)
        {
            sg->wrBuff.vertices[vt] = q;
            sg->wrBuff.labels[vt] = R;
        }
        vt++;

        // Perform the intersection again to find vertices for new subgraph
        for (st = so->st, en = so->en; st < en; st += 32)
        {
            unsigned int i = st + LANEID;
            pred = false;
            if (i < en)
            {
                v = sg->rdBuff.vertices[i];
                label = sg->rdBuff.labels[i];
                // no need to intersect R nodes
                pred = label == R || binarySearch(dp->neighbors, qst, qen, v);
            }

            auto loc = scanIndex(pred) + vt;
            if (pred)
            {
                sg->wrBuff.vertices[loc] = v;
                sg->wrBuff.labels[loc] = label;
                if (label == Q)
                    sg->wrBuff.labels[loc] = v < q ? X : P;
            }

            if(LANEID==31)
                vt = loc + pred;
            vt = __shfl_sync(FULL, vt, 31);
        }
    }

    // this version is called when subgraphs are spawned from Q nodes
    __device__ void
    generateSubgraphs(SubgraphOffsets *so, unsigned int q)
    {
        // let's find expected subgraph length...
        unsigned int sglen = min((unsigned int) (so->en - so->st), dp->neighbors_offset[q + 1] - dp->neighbors_offset[q]);

        // this subgraph might not fit into temp area, so go for double intersection option
        if (sglen > TEMPSIZE)
        {
            generateSubgraphDoubleIntersect(so, q);
            return;
        }

        // else the subgraph can be intersected just once and put it to temp for later reading
        sglen = getSubgraphTemp(so, q);
        if (sglen == 0)
            return; // q doesn't have graph to spawn.
        // sglen = |N(q)∩(XUPUR)|
        // adding 1 in sglen, as q itself appears in subgraph as R
        assert(sglen + 1 < TEMPSIZE);
        // allocates a subgraph by atomic operations, and puts q in subgraph as well
        auto vt = sg->append(sglen + 1);
        if (sg->isOverflow())
            return;
        if (LANEID == 0)
        {
            sg->wrBuff.vertices[vt] = q;
            sg->wrBuff.labels[vt] = R;
        }
        vt++; // as one element is written i.e. q
        unsigned int *tempv = this->tempv + GLWARPID * TEMPSIZE;
        char *templ = this->templ + GLWARPID * TEMPSIZE;

        // subgraph is already stored in temp. q is already written to subgraph
        for (unsigned int i = LANEID; i < sglen; i += 32)
        {
            auto k = vt + i;
            unsigned int v = tempv[i];
            char label = templ[i];
            sg->wrBuff.vertices[k] = v;
            sg->wrBuff.labels[k] = label;
            if (label == Q)
                sg->wrBuff.labels[k] = v < q ? X : P;
        }
    }

    __device__ unsigned int selectPivot(SubgraphOffsets *so)
    {
        unsigned int st = so->st;
        unsigned int en = so->en;
        unsigned int max = 0, pivot;
        bool pred;
        unsigned int plen = 0;

        // Let's write P set to temp location
        for (unsigned int i = st; i < en; i += 32)
        {
            unsigned int il = i + LANEID;
            pred = il < en && sg->rdBuff.labels[il] == P;            // exploiting short-circuit of &&
            plen = writeToTemp(sg->rdBuff.vertices[il], pred, plen); // the function returns update value of plen
        }

        for (unsigned int j = st; j < en; j++)
        {
            // entire warp is processing one element in this loop, hence laneid is not added...
            // it's not a divergence, entire warp will continue as result of below condition
            if (sg->rdBuff.labels[j] == R)
                continue;                            // pivot is selected from P U X
            unsigned int v = sg->rdBuff.vertices[j]; // v ∈ (P U X)
            // (st1, en1) are N(v)
            unsigned int st1 = dp->neighbors_offset[v];
            unsigned int en1 = dp->neighbors_offset[v + 1];
            unsigned int nmatched = 0;
            for (unsigned int k = 0; k < plen; k += 32)
            {
                unsigned int kl = k + LANEID; // need to run all lanes, so that ballot function works well
                pred = kl < plen && binarySearch(dp->neighbors, st1, en1, tempv[kl + GLWARPID * TEMPSIZE]);
                nmatched += __popc(__ballot_sync(FULL, pred));
            }
            if (nmatched >= max) // using == just to take care of case when nmatched is zero for all v
            {
                max = nmatched;
                pivot = v;
            }
        }
        return pivot;
    }

    __device__ void generateSubgraphs(unsigned int base)
    {
        for (unsigned int i = GLWARPID; i < sg->chunk[0]; i += N_WARPS)
        {

            unsigned int v = base + i;
            if (v >= dp->V)
                break;
            unsigned int st = dp->neighbors_offset[v];
            unsigned int en = dp->neighbors_offset[v + 1];
            unsigned int sglen = en - st;
            if (sglen == 0)
                continue; // there was no neighbor for this vertex...
            // adding 1 as vertices in new graph are number of neighbors + v itself
            unsigned int vt = sg->append(sglen + 1); // allocates a subgraph by atomic operations, and puts v as well
            if (LANEID == 0)
            {
                sg->wrBuff.vertices[vt] = v;
                sg->wrBuff.labels[vt] = R;
            }
            vt++; // as one element is written i.e. v
            for (unsigned int j = st + LANEID, k = vt + LANEID; j < en; j += 32, k += 32)
            {
                unsigned int u = dp->neighbors[j];
                sg->wrBuff.vertices[k] = u;
                sg->wrBuff.labels[k] = (u < v) ? X : P;
            }
        }
    }

    __device__ void loadFromHost()
    {
        for (ull k = GLWARPID; k < HOSTCHUNK; k += N_WARPS){
        // while(true)
        // {
        //     unsigned int counter;
        //     if(LANEID==0){
        //         counter = atomicAdd(sgHost->head, 1);
        //     }
        //     counter = __shfl_sync(FULL, counter, 0);
        //     if(counter>=HOSTCHUNK)
        //         break;

            auto so = sgHost->next();
            if (sgHost->isEnd(so))
                break;
            ull sglen = so.en - so.st;
            Index vt = sg->append(sglen);
            for (ull i = so.st + LANEID, j = vt + LANEID; i < so.en; i += 32, j += 32)
            {
                sg->wrBuff.vertices[j] = sgHost->rdBuff.vertices[i];
                sg->wrBuff.labels[j] = sgHost->rdBuff.labels[i];
            }
        }
    }

    __device__ void dumpToHost(SubgraphOffsets *so)
    {
        if(sgHost->overflow[0]){
            // host overflow occured, return from here and end the program by worker
            return;
        }
        ull vt = sgHost->append(so->en - so->st);
        // if(GTHID==0)
        //     printf("%d-%d ", vt, so->en - so->st);

        for (ull i = vt + LANEID, j = so->st + LANEID; j < so->en; i += 32, j += 32)
        {
            sgHost->wrBuff.vertices[i] = sg->rdBuff.vertices[j];
            sgHost->wrBuff.labels[i] = sg->rdBuff.labels[j];
        }
    }
};
class BKExpandSequential : public BKBase
{
    
public:

    __device__ void processSubgraphs()
    {
        while (true)
        {
            // each warp gets a vertex to process. First lane increments, and then s is broadcasted to every lane
            SubgraphOffsets so = sg->next();
            
            if (sg->isEnd(so))
                break;

            if (sg->isOverflow() )
            {
                break;
            }

            if (examineClique(&so))
            {
                // todo report clique
                // ?? do we need to save clique, or just increment a count
                // seemingly GPU-BK(TPDS) is only calculating number of cliques
                if (LANEID == 0)
                {
                    atomicAdd(this->iterCliques, 1);
                }
            }
            else 
            // if (!crossed(&so)) // optional, saves the effort of selectPivot for crossed subgraphs, however adds cost for every other
            {
                if (sg->isOverflowToHost())
                {
                    dumpToHost(&so);
                    continue;
                }

                unsigned int pivot = selectPivot(&so);
                expandClique(&so, pivot);
            }
        }
    }

    __device__ void expandClique(SubgraphOffsets *so, unsigned int pivot)
    {

        unsigned int st = so->st;
        unsigned int en = so->en;
        unsigned int pst = dp->neighbors_offset[pivot];
        unsigned int pen = dp->neighbors_offset[pivot + 1];
        // subgraph stored in (st, en)
        // N(pivot) are in (pst, pen)
        // find Q=P-N(pivot)
        // for every q ∈ Q, generate a subgraph
        for (unsigned int i = st + LANEID; i < en; i += 32)
        {
            if (sg->rdBuff.labels[i] == P && !binarySearch(dp->neighbors, pst, pen, sg->rdBuff.vertices[i]))
            {
                // v belongs to Q, so generate subgraph for it
                // simply change their labels to Q, afterwards generate a subgraph for each such node
                sg->rdBuff.labels[i] = Q;
            }
        }
        // now generate subgraphs for all v's which were put to Q
        // entire warp generates a subgraph one by one for each Q
        for (unsigned int i = st; i < en; i++)
        {
            if (sg->rdBuff.labels[i] == Q)
            {
                generateSubgraphs(so, sg->rdBuff.vertices[i]);
            }
        }
    }

    __device__ void expand()
    {
        // do nothing...
    }
};