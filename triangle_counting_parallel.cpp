#include <iostream>
#include <cstdio>
#include "core/utils.h"
#include "core/graph.h"
#include <mpi.h>

using namespace std;
struct return_val
{
  long triangle;
  double time;
  uintE num_edges;
  int rank;
};



long countTriangles(uintV *array1, uintE len1, uintV *array2, uintE len2,
                    uintV u, uintV v)
{
  uintE i = 0, j = 0; // indexes for array1 and array2
  long count = 0;

  if (u == v)
    return count;

  while ((i < len1) && (j < len2))
  {
    if (array1[i] == array2[j])
    {
      if ((array1[i] != u) && (array1[i] != v))
      {
        count++;
      }
      else
      {
        // triangle with self-referential edge -> ignore
      }
      i++;
      j++;
    }
    else if (array1[i] < array2[j])
    {
      i++;
    }
    else
    {
      j++;
    }
  }
  return count;
}
void triangleCountSerial(Graph &g) {
  uintV n = g.n_;
  long triangle_count = 0;
  double time_taken = 0.0;
  timer t1;
  t1.start();
  for (uintV u = 0; u < n; u++) {
    uintE out_degree = g.vertices_[u].getOutDegree();
    for (uintE i = 0; i < out_degree; i++) {
      uintV v = g.vertices_[u].getOutNeighbor(i);
      triangle_count += countTriangles(g.vertices_[u].getInNeighbors(),
                                       g.vertices_[u].getInDegree(),
                                       g.vertices_[v].getOutNeighbors(),
                                       g.vertices_[v].getOutDegree(), u, v);
    }
  }
  time_taken = t1.stop();
  std::cout << "Number of triangles : " << triangle_count << "\n";
  std::cout << "Number of unique triangles : " << triangle_count / 3 << "\n";
  std::cout << "Time taken (in seconds) : " << std::setprecision(TIME_PRECISION)
            << time_taken << "\n";
}
void EdgeDecompostion(uintV&startV,uintV&endV,Graph&g,uintV n,int world_size,int world_rank){

  uintV start_vertex = 0;
  uintV end_vertex = 0;
  uintE m = g.m_;

  for (int i = 0; i < world_size; i++) {
      start_vertex = end_vertex;
      long count = 0;
      while (end_vertex < n)
      {
       
          count += g.vertices_[end_vertex].getOutDegree();
          end_vertex += 1;
          if (count >= m / world_size)
              break;
      }
      if(i == world_rank)
        break;
      
  }
  startV = start_vertex;
    endV   = end_vertex;
}


return_val triangleCountLocal(Graph &g, uintV start, uintV end, vector<uintE> &edges, uintV n, int rank)
{
  timer t1;
  t1.start();
  auto lower = lower_bound(edges.begin(), edges.end(), start);
  uintV u_;
  if (lower == edges.begin())
  {
    u_ = 0;
  }
  else
  {
    u_ = lower - edges.begin() - 1;
  }
  uintV edge_start = start - edges[u_];

  // thread_id, num_vertices, num_edges, triangle_count, time_taken
  long triangle_count = 0;
  uintE num_edges = 0;
  uintV num_vertices = 0;
  uintV temp_start = start;

  for (uintV u = start; u < end; u++) {
    uintE out_degree = g.vertices_[u].getOutDegree();
    for (uintE i = 0; i < out_degree; i++) {
      uintV v = g.vertices_[u].getOutNeighbor(i);
      triangle_count += countTriangles(g.vertices_[u].getInNeighbors(),
                                       g.vertices_[u].getInDegree(),
                                       g.vertices_[v].getOutNeighbors(),
                                       g.vertices_[v].getOutDegree(), u, v);
      num_edges++;
    }
  }
  double time_taken = t1.stop();
  return_val r = {triangle_count, time_taken, num_edges, rank};
  return r;
}

void MPI_Call(const int world_rank, int world_size, Graph &g)
{
  // do the local work first then send info

  uintV n = g.n_;
  long triangle_count = 0;
  double time_taken = 0.0;
  timer t1;
  t1.start();
  vector<uintE> edges(n + 1, 0);
  uintE edge_count = 0;
  
  uintV nums_per_thread = edge_count / world_size;
  uintV start;
  uintV end; 
  EdgeDecompostion(start,end,g,n,world_size,world_rank);
  // i have the start and end
  return_val local_result = triangleCountLocal(g, start, end, edges, n, world_rank);

  // like the thread part
  if (world_rank == 0)
  {

    vector<return_val> results(world_size);
    results[0] = local_result;
    triangle_count = local_result.triangle;
    for (int non_root = 1; non_root < world_size; non_root++)
    {
      MPI_Recv(&results[non_root], sizeof(return_val), MPI_BYTE, non_root, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      triangle_count += results[non_root].triangle;
    }
    cout << "rank, edges, triangle_count, communication_time" << endl;
    for (int i = 1; i < world_size; i++)
    {
      for (int h = 1; h < world_size; h++)
      {
        if (i == results[h].rank)
          std::cout << i << ", " << results[h].num_edges << ", " << results[h].triangle << ", " << results[h].time << std::endl;
      }
    }
    std::cout << 0 << ", " << results[0].num_edges << ", " << results[0].triangle << ", " << results[0].time << std::endl;
  }
  else
  {
    MPI_Send(&local_result, sizeof(return_val), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
  }

  time_taken = t1.stop();

  // Print out overall statistics
  if (world_rank == 0)
  {
    std::printf("Number of triangles : %ld\n", triangle_count);
    std::printf("Number of unique triangles : %ld\n", triangle_count / 3);
    std::printf("Time taken (in seconds) : %f\n", time_taken);
  }
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  cxxopts::Options options("triangle_counting_serial", "Count the number of triangles using serial and parallel execution");
  options.add_options("custom", {
                                    {"strategy", "Strategy to be used", cxxopts::value<uint>()->default_value(DEFAULT_STRATEGY)},
                                    {"inputFile", "Input graph file path", cxxopts::value<std::string>()->default_value("/scratch/input_graphs/roadNet-CA")},
                                });

  auto cl_options = options.parse(argc, argv);
  uint strategy = cl_options["strategy"].as<uint>();
  std::string input_file_path = cl_options["inputFile"].as<std::string>();
  Graph g;
  g.readGraphFromBinary<int>(input_file_path);
  if (world_rank == 0 && strategy == 1)
  {
    std::printf("World size : %d\n", world_size);
    std::printf("Communication strategy : %d\n", strategy);
  }

  // Get the world size and print it out here
  switch (strategy)
  {
  case 0:
    triangleCountSerial(g);
    break;
  case 1:
    MPI_Call(world_rank, world_size, g);
    break;
  
  default:
    break;
  }

 


  
  // triangleCountSerial(g);
  MPI_Finalize();
  return 0;
}
