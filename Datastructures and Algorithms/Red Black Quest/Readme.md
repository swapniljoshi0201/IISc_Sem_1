Implement the Red-Black Tree data structure that supports the following operations:
Insert – Insert a new element into the tree while maintaining the Red-Black Tree properties. Ensure the tree is rebalanced and recolored as necessary after each insertion.

Delete – Remove an element from the tree while maintaining the Red-Black Tree properties. After deletion, the tree should be rebalanced and recolored as required.

Search – Check if a given element exists in the tree and return a boolean value indicating its presence.

Traversal – Implement the following tree traversal operations:

Inorder Traversal – Return the elements in sorted order.
Preorder Traversal – Return the elements in pre-order.
Postorder Traversal – Return the elements in post-order.
Level-order Traversal – Return the elements level by level.
Input Format

The input consists of:

An integer indicating the operation to perform, as described below:
1 → Insert
2 → Delete
3 → Search
4 → Traversal
An integer representing the number of elements in the input array.
An array of integers (used as input for the selected operation).
An integer element for search operation if operation is search, element for insertion if operation is insert and set to "None" if not applicable.
An integer element for deletion operation and set to "None" if not applicable.
A string indicating the type of traversal (used only for Traversal operation). Possible values:
"inorder"
"preorder"
"postorder"
"levelorder"
Set to "None" if not applicable.
Constraints

Time complexity requirements:

insert – O(log n)
delete – O(log n)
search – O(log n)
traversal – O(n)
Input array size: up to 10⁶ elements.

Element range: [-10⁹, 10⁹].
Do not use built-in balanced tree libraries (e.g., bisect, set, dict, or any external library like bintrees, blist, etc.).
Output Format

The output returns either a boolean value or an array of integers, depending on the operation, as described below:

Insert Operation: Returns an array of integers representing the in-order traversal of the Red-Black Tree after all insertions, maintaining Red-Black Tree properties.

Delete Operation: Returns an array of integers representing the in-order traversal of the Red-Black Tree after all deletions, maintaining Red-Black Tree properties.

Search Operation: Returns a single boolean value — True if the element is found, False otherwise.

Traversal Operation: Returns an array of integers based on the specified traversal type:

"inorder" → Elements in ascending order.
"preorder" → Elements in pre-order sequence.
"postorder" → Elements in post-order sequence.
"levelorder" → Elements level by level.
Sample Input 0

1
5
20 15 25 10 18
None
None
None
Sample Output 0

10 15 18 20 25
Sample Input 1

3
3
15 10 20
3
None
None
Sample Output 1

False
Sample Input 2

2
2
50 70
None
50
None
Sample Output 2

70
Sample Input 3

4
5
10 20 30 40 50
None
None
inorder
Sample Output 3

10 20 30 40 50