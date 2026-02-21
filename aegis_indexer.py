import numpy as np

class KDNode:
    """A node in the K-Dimensional Tree."""
    def __init__(self, vector, payload, left=None, right=None, axis=0):
        self.vector = vector
        self.payload = payload  # Contains {"response": str, "tokens": list}
        self.left = left
        self.right = right
        self.axis = axis

class RBAC_KDTree:
    """
    A custom KD-Tree implementation optimized for L2-normalized vectors.
    Search time complexity: O(log N) instead of O(N).
    """
    def __init__(self, k_dimensions=128):
        self.root = None
        self.k = k_dimensions

    def insert(self, vector: np.ndarray, payload: dict):
        """Inserts a new prompt/response into the tree."""
        if self.root is None:
            self.root = KDNode(vector, payload, axis=0)
            return

        node = self.root
        while True:
            axis = node.axis
            next_axis = (axis + 1) % self.k
            
            if vector[axis] < node.vector[axis]:
                if node.left is None:
                    node.left = KDNode(vector, payload, axis=next_axis)
                    break
                node = node.left
            else:
                if node.right is None:
                    node.right = KDNode(vector, payload, axis=next_axis)
                    break
                node = node.right
    

    def visualize_tree(self):
        """Prints a visual representation of the mathematical KD-Tree to the console."""
        def print_tree(node, level=0, prefix="Root: "):
            if node is not None:
                # Print current node's prompt and which axis it split on
                indent = "    " * level
                print(f"{indent}{prefix}[Axis {node.axis}] {node.payload['prompt']}")
                
                # Recursively print left and right mathematical branches
                if node.left or node.right:
                    if node.left:
                        print_tree(node.left, level + 1, "├── Left (< Median): ")
                    else:
                        print(f"{indent}    ├── Left: [Empty Space]")
                        
                    if node.right:
                        print_tree(node.right, level + 1, "└── Right (>= Median): ")
                    else:
                        print(f"{indent}    └── Right: [Empty Space]")

        print("\n=== KD-Tree Mathematical Spatial Map ===")
        if self.root is None:
            print("Tree is empty.")
        else:
            print_tree(self.root)
        print("========================================\n")
        

    def nearest_neighbor(self, target_vector: np.ndarray, threshold_distance: float):
        """
        Traverses the tree to find the closest semantic vector.
        Uses hypersphere intersection to prune (ignore) irrelevant branches.
        """
        best_node = None
        best_dist = float('inf')

        def _search(node):
            nonlocal best_node, best_dist
            if node is None:
                return

            # Calculate squared Euclidean distance (faster than computing square root)
            dist = np.sum((node.vector - target_vector) ** 2)
            
            if dist < best_dist:
                best_dist = dist
                best_node = node

            # Determine which branch to search first based on the splitting axis
            axis = node.axis
            diff = target_vector[axis] - node.vector[axis]

            close_branch = node.left if diff < 0 else node.right
            far_branch = node.right if diff < 0 else node.left

            # 1. Always search the side the target point falls on
            _search(close_branch)

            # 2. Mathematical Pruning: 
            # Only search the opposite side if the distance to the splitting plane 
            # is LESS than the current best distance.
            if (diff ** 2) < best_dist:
                _search(far_branch)

        _search(self.root)
        
        # If the closest found node is within our acceptable similarity threshold, return it
        if best_dist <= threshold_distance:
            return best_node
        return None
