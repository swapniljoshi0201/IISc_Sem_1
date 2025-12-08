Routefinder — Shortest Routes with Dijkstra’s Min-Heap
You are given a directed weighted graph ( G = (V, E) ) and a source vertex S. Your task is to determine the shortest path length from the source vertex S to all other vertices in the graph.

All edge weights are non-negative. To efficiently select the next vertex with the smallest tentative distance, you must use a Min-Heap.

Input Format

The input consists of:

An integer V, the number of vertices in the graph.
An integer E, the number of edges.
E lines follow; each line contains three integers:
u — start vertex of the edge
v — end vertex of the edge
w — weight of the edge from u to v
An integer S, the source vertex.
Vertices are numbered from 1 to V.

Constraints

1 ≤ V ≤ 105
1 ≤ E ≤ 2 × 105
1 ≤ u, v ≤ V
0 ≤ w ≤ 109
Graph may contain multiple edges and self-loops.
All edge weights are non-negative.
Output Format

Print V space-separated values — the shortest distances from the source vertex S to vertices 1 through V.

For reachable vertices print the distance (integer).
For vertices not reachable from S, print -1.
Sample Input 0

5 7
1 2 2
1 3 4
2 3 1
2 4 7
3 5 3
4 5 1
5 4 2
1
Sample Output 0

0 2 3 8 6
Sample Input 1

4 3
1 2 5
2 3 3
3 4 1
2
Sample Output 1

-1 0 3 4