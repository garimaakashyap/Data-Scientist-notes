# Python Data Structures & Algorithms

## Table of Contents
1. [Arrays and Lists](#arrays-and-lists)
2. [Linked Lists](#linked-lists)
3. [Stacks and Queues](#stacks-and-queues)
4. [Trees](#trees)
5. [Graphs](#graphs)
6. [Hash Tables and Dictionaries](#hash-tables-and-dictionaries)
7. [Heaps and Priority Queues](#heaps-and-priority-queues)
8. [Sorting Algorithms](#sorting-algorithms)
9. [Searching Algorithms](#searching-algorithms)
10. [Dynamic Programming](#dynamic-programming)
11. [Greedy Algorithms](#greedy-algorithms)
12. [Graph Algorithms](#graph-algorithms)
13. [String Algorithms](#string-algorithms)
14. [Common Patterns](#common-patterns)
15. [Python-Specific Optimizations](#python-specific-optimizations)

---

## Arrays and Lists

### Basic Operations
```python
# Time complexities
# Access: O(1)
# Search: O(n)
# Insertion: O(n)
# Deletion: O(n)

arr = [1, 2, 3, 4, 5]

# Access
arr[0]  # O(1)

# Search
3 in arr  # O(n)

# Insertion
arr.append(6)        # O(1) amortized
arr.insert(0, 0)     # O(n)
arr.extend([7, 8])   # O(k) where k is length of extension

# Deletion
arr.remove(3)        # O(n)
arr.pop()            # O(1)
arr.pop(0)           # O(n)
del arr[0]           # O(n)
```

### Two Pointer Technique
```python
# Reverse array
def reverse_array(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1
    return arr

# Two sum (sorted array)
def two_sum_sorted(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []

# Remove duplicates (sorted array)
def remove_duplicates(arr):
    if not arr:
        return 0
    write_index = 1
    for read_index in range(1, len(arr)):
        if arr[read_index] != arr[read_index - 1]:
            arr[write_index] = arr[read_index]
            write_index += 1
    return write_index
```

### Sliding Window
```python
# Maximum sum of subarray of size k
def max_sum_subarray(arr, k):
    if len(arr) < k:
        return 0
    
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

# Longest substring without repeating characters
def longest_unique_substring(s):
    char_map = {}
    left = 0
    max_len = 0
    
    for right in range(len(s)):
        if s[right] in char_map and char_map[s[right]] >= left:
            left = char_map[s[right]] + 1
        char_map[s[right]] = right
        max_len = max(max_len, right - left + 1)
    
    return max_len
```

---

## Linked Lists

### Singly Linked List
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, val):
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def prepend(self, val):
        new_node = ListNode(val)
        new_node.next = self.head
        self.head = new_node
    
    def delete(self, val):
        if not self.head:
            return
        if self.head.val == val:
            self.head = self.head.next
            return
        current = self.head
        while current.next:
            if current.next.val == val:
                current.next = current.next.next
                return
            current = current.next
    
    def display(self):
        values = []
        current = self.head
        while current:
            values.append(current.val)
            current = current.next
        return values

# Reverse linked list
def reverse_linked_list(head):
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev

# Detect cycle
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

# Merge two sorted lists
def merge_sorted_lists(l1, l2):
    dummy = ListNode(0)
    current = dummy
    
    while l1 and l2:
        if l1.val <= l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    
    current.next = l1 or l2
    return dummy.next
```

### Doubly Linked List
```python
class DoublyListNode:
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    
    def append(self, val):
        new_node = DoublyListNode(val)
        if not self.head:
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node
```

---

## Stacks and Queues

### Stack Implementation
```python
# Using list (Python's list is already a stack)
stack = []
stack.append(1)  # Push
stack.append(2)
stack.append(3)
top = stack.pop()  # Pop (returns 3)
peek = stack[-1]   # Peek

# Custom stack class
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        return None
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        return None
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

# Valid parentheses
def is_valid_parentheses(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    
    return not stack

# Next greater element
def next_greater_element(nums):
    stack = []
    result = [-1] * len(nums)
    
    for i in range(len(nums)):
        while stack and nums[stack[-1]] < nums[i]:
            result[stack.pop()] = nums[i]
        stack.append(i)
    
    return result
```

### Queue Implementation
```python
# Using collections.deque (recommended)
from collections import deque

queue = deque()
queue.append(1)  # Enqueue
queue.append(2)
queue.append(3)
front = queue.popleft()  # Dequeue (returns 1)

# Custom queue class
class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        if not self.is_empty():
            return self.items.popleft()
        return None
    
    def front(self):
        if not self.is_empty():
            return self.items[0]
        return None
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

# Circular queue
class CircularQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = [None] * capacity
        self.front = 0
        self.rear = -1
        self.size = 0
    
    def enqueue(self, item):
        if self.is_full():
            return False
        self.rear = (self.rear + 1) % self.capacity
        self.queue[self.rear] = item
        self.size += 1
        return True
    
    def dequeue(self):
        if self.is_empty():
            return None
        item = self.queue[self.front]
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return item
    
    def is_empty(self):
        return self.size == 0
    
    def is_full(self):
        return self.size == self.capacity
```

### Priority Queue
```python
import heapq

# Min heap (default)
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 2)
min_val = heapq.heappop(heap)  # 1

# Max heap (negate values)
heap = []
heapq.heappush(heap, -3)
heapq.heappush(heap, -1)
heapq.heappush(heap, -2)
max_val = -heapq.heappop(heap)  # 3

# Heap from list
nums = [3, 1, 4, 1, 5, 9, 2, 6]
heapq.heapify(nums)  # Convert to heap in-place

# K largest elements
def k_largest(nums, k):
    return heapq.nlargest(k, nums)

# K smallest elements
def k_smallest(nums, k):
    return heapq.nsmallest(k, nums)
```

---

## Trees

### Binary Tree
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Tree traversals
def preorder_traversal(root):
    if not root:
        return []
    return [root.val] + preorder_traversal(root.left) + preorder_traversal(root.right)

def inorder_traversal(root):
    if not root:
        return []
    return inorder_traversal(root.left) + [root.val] + inorder_traversal(root.right)

def postorder_traversal(root):
    if not root:
        return []
    return postorder_traversal(root.left) + postorder_traversal(root.right) + [root.val]

# Level-order traversal (BFS)
from collections import deque

def level_order_traversal(root):
    if not root:
        return []
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    
    return result

# Maximum depth
def max_depth(root):
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))

# Same tree
def is_same_tree(p, q):
    if not p and not q:
        return True
    if not p or not q:
        return False
    return (p.val == q.val and 
            is_same_tree(p.left, q.left) and 
            is_same_tree(p.right, q.right))

# Symmetric tree
def is_symmetric(root):
    def is_mirror(left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        return (left.val == right.val and 
                is_mirror(left.left, right.right) and 
                is_mirror(left.right, right.left))
    
    if not root:
        return True
    return is_mirror(root.left, root.right)
```

### Binary Search Tree
```python
class BSTNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BST:
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        self.root = self._insert(self.root, val)
    
    def _insert(self, root, val):
        if not root:
            return BSTNode(val)
        if val < root.val:
            root.left = self._insert(root.left, val)
        else:
            root.right = self._insert(root.right, val)
        return root
    
    def search(self, val):
        return self._search(self.root, val)
    
    def _search(self, root, val):
        if not root or root.val == val:
            return root
        if val < root.val:
            return self._search(root.left, val)
        return self._search(root.right, val)
    
    def delete(self, val):
        self.root = self._delete(self.root, val)
    
    def _delete(self, root, val):
        if not root:
            return root
        if val < root.val:
            root.left = self._delete(root.left, val)
        elif val > root.val:
            root.right = self._delete(root.right, val)
        else:
            if not root.left:
                return root.right
            elif not root.right:
                return root.left
            root.val = self._min_value(root.right)
            root.right = self._delete(root.right, root.val)
        return root
    
    def _min_value(self, root):
        while root.left:
            root = root.left
        return root.val

# Validate BST
def is_valid_bst(root, min_val=float('-inf'), max_val=float('inf')):
    if not root:
        return True
    if root.val <= min_val or root.val >= max_val:
        return False
    return (is_valid_bst(root.left, min_val, root.val) and 
            is_valid_bst(root.right, root.val, max_val))
```

---

## Graphs

### Graph Representation
```python
# Adjacency list
from collections import defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
    
    def add_edge(self, u, v):
        self.graph[u].append(v)
        # For undirected graph, also add:
        # self.graph[v].append(u)
    
    def dfs(self, start):
        visited = set()
        result = []
        
        def dfs_helper(node):
            visited.add(node)
            result.append(node)
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    dfs_helper(neighbor)
        
        dfs_helper(start)
        return result
    
    def bfs(self, start):
        visited = set()
        queue = deque([start])
        result = []
        
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                result.append(node)
                for neighbor in self.graph[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        
        return result

# DFS iterative
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    result = []
    
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            result.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return result
```

### Shortest Path
```python
from collections import deque

# BFS for unweighted graph
def shortest_path_bfs(graph, start, end):
    if start == end:
        return [start]
    
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        node, path = queue.popleft()
        for neighbor in graph[node]:
            if neighbor == end:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None

# Dijkstra's algorithm (weighted graph)
import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)
        
        for neighbor, weight in graph[current]:
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances
```

---

## Hash Tables and Dictionaries

### Hash Table Operations
```python
# Python dict is a hash table
# Average time: O(1) for insert, delete, search
# Worst case: O(n)

hash_table = {}

# Insert
hash_table['key'] = 'value'  # O(1) average

# Search
value = hash_table.get('key')  # O(1) average
value = hash_table['key']      # O(1) average, KeyError if not found

# Delete
del hash_table['key']  # O(1) average
hash_table.pop('key')  # O(1) average

# Two sum problem
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# Group anagrams
def group_anagrams(strs):
    from collections import defaultdict
    groups = defaultdict(list)
    for s in strs:
        key = ''.join(sorted(s))
        groups[key].append(s)
    return list(groups.values())
```

---

## Heaps and Priority Queues

### Heap Operations
```python
import heapq

# Min heap
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 2)
min_val = heapq.heappop(heap)  # 1

# Max heap (negate values)
max_heap = []
heapq.heappush(max_heap, -3)
heapq.heappush(max_heap, -1)
max_val = -heapq.heappop(max_heap)  # 3

# Heap from list
nums = [3, 1, 4, 1, 5, 9, 2, 6]
heapq.heapify(nums)  # O(n) - converts list to heap in-place

# K largest elements
def k_largest(nums, k):
    return heapq.nlargest(k, nums)

# Merge k sorted lists
def merge_k_sorted_lists(lists):
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))
    
    result = []
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        if elem_idx + 1 < len(lists[list_idx]):
            heapq.heappush(heap, (lists[list_idx][elem_idx + 1], 
                                 list_idx, elem_idx + 1))
    return result
```

---

## Sorting Algorithms

### Bubble Sort
```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
# Time: O(n²), Space: O(1)
```

### Selection Sort
```python
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
# Time: O(n²), Space: O(1)
```

### Insertion Sort
```python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
# Time: O(n²), Space: O(1)
```

### Merge Sort
```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
# Time: O(n log n), Space: O(n)
```

### Quick Sort
```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# In-place version
def quicksort_inplace(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    if low < high:
        pivot_idx = partition(arr, low, high)
        quicksort_inplace(arr, low, pivot_idx - 1)
        quicksort_inplace(arr, pivot_idx + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
# Time: O(n log n) average, O(n²) worst, Space: O(log n)
```

### Heap Sort
```python
import heapq

def heap_sort(arr):
    heap = []
    for val in arr:
        heapq.heappush(heap, val)
    return [heapq.heappop(heap) for _ in range(len(heap))]
# Time: O(n log n), Space: O(n)
```

---

## Searching Algorithms

### Linear Search
```python
def linear_search(arr, target):
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1
# Time: O(n), Space: O(1)
```

### Binary Search
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Recursive version
def binary_search_recursive(arr, target, left=0, right=None):
    if right is None:
        right = len(arr) - 1
    if left > right:
        return -1
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
# Time: O(log n), Space: O(1) iterative, O(log n) recursive
```

---

## Dynamic Programming

### Fibonacci
```python
# Naive recursive (O(2^n))
def fib_naive(n):
    if n <= 1:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)

# Memoization (O(n))
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]

# Tabulation (O(n))
def fib_tabulation(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

# Space-optimized (O(1))
def fib_optimized(n):
    if n <= 1:
        return n
    prev2, prev1 = 0, 1
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    return prev1
```

### Climbing Stairs
```python
def climb_stairs(n):
    if n <= 2:
        return n
    prev2, prev1 = 1, 2
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    return prev1
```

### Longest Common Subsequence
```python
def lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]
```

### Coin Change
```python
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1
```

---

## Greedy Algorithms

### Activity Selection
```python
def activity_selection(activities):
    # activities is list of (start, end) tuples
    activities.sort(key=lambda x: x[1])  # Sort by end time
    selected = [activities[0]]
    
    for activity in activities[1:]:
        if activity[0] >= selected[-1][1]:
            selected.append(activity)
    
    return selected
```

### Fractional Knapsack
```python
def fractional_knapsack(items, capacity):
    # items is list of (value, weight) tuples
    items.sort(key=lambda x: x[0] / x[1], reverse=True)
    total_value = 0
    
    for value, weight in items:
        if capacity >= weight:
            total_value += value
            capacity -= weight
        else:
            total_value += value * (capacity / weight)
            break
    
    return total_value
```

---

## Graph Algorithms

### Topological Sort
```python
def topological_sort(graph):
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1
    
    queue = deque([node for node in in_degree if in_degree[node] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == len(graph) else []
```

### Union-Find (Disjoint Set)
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
```

---

## String Algorithms

### KMP Algorithm
```python
def kmp_search(text, pattern):
    def build_lps(pattern):
        lps = [0] * len(pattern)
        length = 0
        i = 1
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps
    
    lps = build_lps(pattern)
    i = j = 0
    result = []
    
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
        if j == len(pattern):
            result.append(i - j)
            j = lps[j - 1]
        elif i < len(text) and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return result
```

### Rabin-Karp Algorithm
```python
def rabin_karp(text, pattern):
    d = 256  # Number of characters in input alphabet
    q = 101  # Prime number
    m = len(pattern)
    n = len(text)
    h = pow(d, m - 1) % q
    
    p = t = 0
    result = []
    
    # Calculate hash value for pattern and first window of text
    for i in range(m):
        p = (d * p + ord(pattern[i])) % q
        t = (d * t + ord(text[i])) % q
    
    # Slide pattern over text one by one
    for i in range(n - m + 1):
        if p == t:
            if text[i:i + m] == pattern:
                result.append(i)
        if i < n - m:
            t = (d * (t - ord(text[i]) * h) + ord(text[i + m])) % q
            if t < 0:
                t = t + q
    
    return result
```

---

## Common Patterns

### Two Pointers
```python
# Used for: sorted arrays, palindromes, two sum
def two_pointers_example(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []
```

### Sliding Window
```python
# Used for: subarrays, substrings, maximum/minimum in window
def sliding_window_example(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum
```

### Fast and Slow Pointers
```python
# Used for: cycle detection, middle of linked list
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

### Merge Intervals
```python
def merge_intervals(intervals):
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        if current[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], current[1])
        else:
            merged.append(current)
    
    return merged
```

---

## Python-Specific Optimizations

### Using Built-in Functions
```python
# Use built-in functions (they're optimized in C)
sum(arr)          # Faster than manual loop
max(arr)          # Faster than manual loop
min(arr)          # Faster than manual loop
all(conditions)   # Short-circuits
any(conditions)   # Short-circuits
```

### List Comprehensions
```python
# Faster than loops
squares = [x**2 for x in range(10)]  # Faster than append in loop
```

### Generators
```python
# Memory efficient
def squares_generator(n):
    for i in range(n):
        yield i**2

# Use when you don't need all values at once
for square in squares_generator(1000000):
    process(square)
```

### Using Sets for Lookups
```python
# O(1) lookup vs O(n) for lists
my_set = set(arr)  # Use set for membership testing
if value in my_set:  # O(1)
    pass
```

### Memoization with functools
```python
from functools import lru_cache

@lru_cache(maxsize=None)
def expensive_function(n):
    # Expensive computation
    return result
```

---

## Practice Exercises

1. Implement a stack using two queues
2. Find the lowest common ancestor in a binary tree
3. Implement Dijkstra's algorithm for shortest path
4. Solve the "Climbing Stairs" problem using DP
5. Implement a Trie data structure
6. Find all anagrams in a string
7. Implement a circular queue
8. Solve the "Coin Change" problem
9. Implement a binary search tree with all operations
10. Find the longest palindromic substring

