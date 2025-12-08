Implement the Heap data structure (Min-Heap) that supports the following operations:

Insert – Add an element to the heap while maintaining the heap property.
Extract Min – Remove and return the minimum element from the heap.
Decrease Key – Decrease the value of a key at a given index and restore the heap property if necessary.
Heapify – Restore the heap property for a subtree rooted at a given index.
Heap Sort - Use your heap implementation to sort an array of integers in ascending order.
Input Format

The input consists of:

An integer indicating the operation to perform, as described below:
1 → Insert
2 → Extract Min
3 → Decrease Key
4 → Heapify
5 → Heap Sort
An integer for number of elements in input array.
An array of integers (used as input to implement operations).
An integer (index for Heapify and Decrease Key operations, set to None if not applicable).
An integer (new key for the Decrease Key operation, set to None if not applicable).
Constraints

Time complexity requirements:

insert – O(log n)
extract_min – O(log n)
decrease_key – O(log n)
heapify – O(log n)
heap_sort – O(n log n)
Input array size: up to 10⁶ elements.

Element range: [-10⁹, 10⁹].
Do not use built-in heap or priority queue libraries.
Output Format

The output returns either an array of integers or a single integer, depending on the operation, as described below:

Insert Operation: Returns an array of integers with the heap structure maintained.
Extract Min Operation: Returns a single integer (the minimum element).
Decrease Key Operation: Returns an array of integers with the heap structure maintained.
Heapify Operation: Returns an array of integers representing the heap structure for a subtree rooted at the given index.
Heap Sort Operation: Returns an array of integers sorted in ascending order.

___________________________________________
Sample Input 0

1
3
3 5 7
None
2
Sample Output 0

2 3 7 5
___________________________________________
Sample Input 1

2
4
2 3 7 5
None
None

Sample Output 1

2
___________________________________________