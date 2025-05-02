import networkx as nx
import csv
import pandas as pd
import os
import pickle
import json


class BusinessRulesGraph:
    """
    A class to build and manage a traversal graph based on business rules from a CSV file.
    Each node represents a condition or decision point, and paths lead to resolution nodes.
    """
    
    def __init__(self, name="business_rules_graph"):
        """
        Initialize a new business rules graph.
        
        Args:
            name (str): Name of the graph
        """
        self.name = name
        self.graph = nx.DiGraph(name=name)
        self.condition_nodes = {}  # Maps condition IDs to node IDs
        self.resolution_nodes = {}  # Maps resolution IDs to node IDs
        
    def load_csv(self, csv_file, condition_cols=None, resolution_col=None, 
                 node_id_col=None, parent_id_col=None):
        """
        Load business rules from a CSV file and construct the graph.
        
        There are two ways this can work:
        1. Tree structure: Each row defines a node with its parent (requires node_id_col and parent_id_col)
        2. Rule paths: Each row defines a complete path of conditions leading to a resolution
        
        Args:
            csv_file (str): Path to the CSV file
            condition_cols (list): Column names that contain conditions (for rule paths)
            resolution_col (str): Column name that contains the resolution (for rule paths)
            node_id_col (str): Column name that contains the node ID (for tree structure)
            parent_id_col (str): Column name that contains the parent node ID (for tree structure)
            
        Returns:
            bool: True if loading was successful
        """
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Determine the type of CSV structure
            if node_id_col and parent_id_col:
                return self._build_from_tree_structure(df, node_id_col, parent_id_col)
            elif condition_cols and resolution_col:
                return self._build_from_rule_paths(df, condition_cols, resolution_col)
            else:
                print("Error: Must provide either (node_id_col and parent_id_col) OR (condition_cols and resolution_col)")
                return False
                
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return False
            
    def _build_from_tree_structure(self, df, node_id_col, parent_id_col):
        """
        Build the graph from a CSV where each row defines a node with its parent.
        
        Args:
            df (DataFrame): Pandas DataFrame containing the data
            node_id_col (str): Column name that contains the node ID
            parent_id_col (str): Column name that contains the parent node ID
            
        Returns:
            bool: True if building was successful
        """
        # Add all nodes first
        for _, row in df.iterrows():
            node_id = row[node_id_col]
            
            # Extract node properties (all columns except node_id and parent_id)
            properties = {col: row[col] for col in df.columns 
                          if col != node_id_col and col != parent_id_col}
            
            # Add node to graph
            self.graph.add_node(node_id, **properties)
            
            # If this is a resolution node (no children), track it
            if 'is_resolution' in properties and properties['is_resolution']:
                self.resolution_nodes[node_id] = node_id
            else:
                self.condition_nodes[node_id] = node_id
                
        # Add edges based on parent-child relationships
        for _, row in df.iterrows():
            node_id = row[node_id_col]
            parent_id = row[parent_id_col]
            
            # Skip if no parent (root node)
            if pd.isna(parent_id):
                continue
                
            # Add edge from parent to child
            self.graph.add_edge(parent_id, node_id)
            
        return True
    
    def _build_from_rule_paths(self, df, condition_cols, resolution_col):
        """
        Build the graph from a CSV where each row defines a complete path of conditions.
        
        Args:
            df (DataFrame): Pandas DataFrame containing the data
            condition_cols (list): Column names that contain conditions
            resolution_col (str): Column name that contains the resolution
            
        Returns:
            bool: True if building was successful
        """
        # Process each row as a path
        for idx, row in df.iterrows():
            prev_node_id = None
            path_nodes = []
            
            # Process each condition column in order
            for i, col in enumerate(condition_cols):
                condition_value = row[col]
                
                # Skip empty conditions
                if pd.isna(condition_value):
                    continue
                
                # Create a unique ID for this condition node
                condition_id = f"{col}:{condition_value}"
                node_id = f"condition_{idx}_{i}_{condition_value}"
                
                # Check if we already have this exact condition
                if condition_id in self.condition_nodes:
                    # Use existing node
                    node_id = self.condition_nodes[condition_id]
                else:
                    # Create new node
                    self.graph.add_node(node_id, 
                                      type='condition', 
                                      attribute=col,
                                      value=condition_value,
                                      description=f"{col} = {condition_value}")
                    self.condition_nodes[condition_id] = node_id
                
                path_nodes.append(node_id)
                
                # Add edge from previous node if exists
                if prev_node_id:
                    self.graph.add_edge(prev_node_id, node_id, order=i)
                
                prev_node_id = node_id
            
            # Add resolution node
            resolution_value = row[resolution_col]
            resolution_id = f"{resolution_col}:{resolution_value}"
            res_node_id = f"resolution_{idx}_{resolution_value}"
            
            if resolution_id in self.resolution_nodes:
                res_node_id = self.resolution_nodes[resolution_id]
            else:
                self.graph.add_node(res_node_id, 
                                  type='resolution',
                                  value=resolution_value,
                                  description=f"Resolution: {resolution_value}")
                self.resolution_nodes[resolution_id] = res_node_id
            
            # Add edge from last condition to resolution
            if prev_node_id:
                self.graph.add_edge(prev_node_id, res_node_id)
                
        return True
    
    def traverse(self, input_data):
        """
        Traverse the graph based on the input data and find the appropriate resolution.
        
        Args:
            input_data (dict): Dictionary of attribute-value pairs
            
        Returns:
            tuple: (resolution_node, path) - Resolution node and path taken
        """
        # Find root nodes (nodes with no incoming edges)
        root_nodes = [n for n, d in self.graph.in_degree() if d == 0]
        
        if not root_nodes:
            return None, []
        
        # Start from each root node and try to find a path
        for root in root_nodes:
            path = [root]
            current_node = root
            
            while True:
                # Get all outgoing edges
                next_nodes = list(self.graph.successors(current_node))
                
                if not next_nodes:
                    # We've reached a leaf node
                    if self.graph.nodes[current_node].get('type') == 'resolution':
                        return current_node, path
                    else:
                        # This path doesn't lead to a resolution
                        break
                
                # Find the next matching node
                found_match = False
                for next_node in next_nodes:
                    node_attrs = self.graph.nodes[next_node]
                    
                    if node_attrs.get('type') == 'resolution':
                        # Found a resolution node
                        path.append(next_node)
                        return next_node, path
                    
                    # Check if this condition matches our input data
                    if node_attrs.get('type') == 'condition':
                        attribute = node_attrs.get('attribute')
                        value = node_attrs.get('value')
                        
                        if attribute in input_data and input_data[attribute] == value:
                            # This condition matches
                            path.append(next_node)
                            current_node = next_node
                            found_match = True
                            break
                
                if not found_match:
                    # No matching condition found
                    break
        
        # No path found
        return None, []
    
    def get_all_paths(self):
        """
        Get all possible paths from condition nodes to resolution nodes.
        
        Returns:
            list: List of paths (each path is a list of node IDs)
        """
        paths = []
        
        # Find all resolution nodes
        resolution_nodes = [n for n, attrs in self.graph.nodes(data=True) 
                            if attrs.get('type') == 'resolution']
        
        # Find all root nodes
        root_nodes = [n for n, d in self.graph.in_degree() if d == 0]
        
        # Find all paths from roots to resolutions
        for root in root_nodes:
            for resolution in resolution_nodes:
                for path in nx.all_simple_paths(self.graph, root, resolution):
                    paths.append(path)
        
        return paths
    
    def visualize(self, output_file=None):
        """
        Generate a visualization of the graph.
        Requires matplotlib and networkx.
        
        Args:
            output_file (str, optional): Path to save the visualization
            
        Returns:
            None
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create node colors
            node_colors = []
            for node in self.graph.nodes():
                if self.graph.nodes[node].get('type') == 'resolution':
                    node_colors.append('lightgreen')
                else:
                    node_colors.append('lightblue')
            
            # Create labels
            labels = {}
            for node in self.graph.nodes():
                attrs = self.graph.nodes[node]
                if 'description' in attrs:
                    labels[node] = attrs['description']
                elif 'attribute' in attrs and 'value' in attrs:
                    labels[node] = f"{attrs['attribute']}={attrs['value']}"
                else:
                    labels[node] = str(node)
            
            # Create position layout
            pos = nx.spring_layout(self.graph)
            
            # Draw the graph
            plt.figure(figsize=(12, 8))
            nx.draw(self.graph, pos, with_labels=False, node_color=node_colors, 
                    node_size=1500, alpha=0.8, arrows=True)
            nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=10)
            
            if output_file:
                plt.savefig(output_file)
            plt.show()
            
        except ImportError:
            print("Visualization requires matplotlib. Install with: pip install matplotlib")
    
    def save(self, filepath=None, format='pickle'):
        """
        Save the graph to a file.
        
        Args:
            filepath (str, optional): Path to save the file
            format (str): Format to save as ('pickle', 'json', or 'graphml')
            
        Returns:
            str: Path to the saved file
        """
        if filepath is None:
            filepath = f"{self.name}.{format}"
            
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'graph': self.graph,
                    'condition_nodes': self.condition_nodes,
                    'resolution_nodes': self.resolution_nodes,
                    'name': self.name
                }, f)
        elif format == 'json':
            # Convert to node-link format
            data = nx.node_link_data(self.graph)
            # Add metadata
            data['condition_nodes'] = self.condition_nodes
            data['resolution_nodes'] = self.resolution_nodes
            data['name'] = self.name
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'graphml':
            # GraphML doesn't support metadata directly
            nx.write_graphml(self.graph, filepath)
            print("Warning: GraphML format doesn't save condition/resolution node mappings")
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        return filepath
    
    def load(self, filepath, format='pickle'):
        """
        Load the graph from a file.
        
        Args:
            filepath (str): Path to the file
            format (str): Format to load from ('pickle', 'json', or 'graphml')
            
        Returns:
            bool: True if loading was successful
        """
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return False
            
        try:
            if format == 'pickle':
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    self.graph = data['graph']
                    self.condition_nodes = data['condition_nodes']
                    self.resolution_nodes = data['resolution_nodes']
                    self.name = data['name']
            elif format == 'json':
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self.graph = nx.node_link_graph(data)
                    if 'condition_nodes' in data:
                        self.condition_nodes = data['condition_nodes']
                    if 'resolution_nodes' in data:
                        self.resolution_nodes = data['resolution_nodes']
                    if 'name' in data:
                        self.name = data['name']
            elif format == 'graphml':
                self.graph = nx.read_graphml(filepath)
                # Rebuild condition/resolution node mappings
                self.condition_nodes = {}
                self.resolution_nodes = {}
                for node, attrs in self.graph.nodes(data=True):
                    if attrs.get('type') == 'condition':
                        key = f"{attrs.get('attribute')}:{attrs.get('value')}"
                        self.condition_nodes[key] = node
                    elif attrs.get('type') == 'resolution':
                        key = f"resolution:{attrs.get('value')}"
                        self.resolution_nodes[key] = node
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def get_node_info(self, node_id):
        """
        Get information about a node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            dict: Node attributes
        """
        if node_id in self.graph.nodes:
            return dict(self.graph.nodes[node_id])
        return None


# Example usage with a sample CSV file
def example_usage():
    """
    Example of how to use the BusinessRulesGraph class.
    """
    # Create a sample CSV file
    with open('business_rules.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Product', 'Region', 'CustomerType', 'Resolution'])
        writer.writerow(['Laptop', 'North', 'Business', 'Premium Support'])
        writer.writerow(['Laptop', 'South', 'Business', 'Standard Support'])
        writer.writerow(['Desktop', 'North', 'Business', 'On-site Support'])
        writer.writerow(['Desktop', 'South', 'Business', 'Remote Support'])
        writer.writerow(['Laptop', 'North', 'Individual', 'Basic Support'])
        writer.writerow(['Laptop', 'South', 'Individual', 'Email Support'])
        writer.writerow(['Desktop', 'North', 'Individual', 'Phone Support'])
        writer.writerow(['Desktop', 'South', 'Individual', 'Self-service'])
    
    # Create a new business rules graph
    graph = BusinessRulesGraph("support_rules")
    
    # Load rules from CSV
    graph.load_csv('business_rules.csv', 
                 condition_cols=['Product', 'Region', 'CustomerType'], 
                 resolution_col='Resolution')
    
    # Print all paths
    print("All possible paths:")
    for path in graph.get_all_paths():
        path_nodes = [graph.get_node_info(node) for node in path]
        path_desc = " -> ".join([node.get('description', str(node)) 
                               for node in path_nodes])
        print(f"  {path_desc}")
    
    # Test traversal with sample inputs
    test_inputs = [
        {'Product': 'Laptop', 'Region': 'North', 'CustomerType': 'Business'},
        {'Product': 'Desktop', 'Region': 'South', 'CustomerType': 'Individual'}
    ]
    
    for input_data in test_inputs:
        resolution_node, path = graph.traverse(input_data)
        
        print(f"\nInput: {input_data}")
        if resolution_node:
            resolution_info = graph.get_node_info(resolution_node)
            print(f"Resolution: {resolution_info.get('value')}")
            
            print("Path taken:")
            for node in path:
                node_info = graph.get_node_info(node)
                print(f"  {node_info.get('description', node)}")
        else:
            print("No matching resolution found")
    
    # Save the graph
    saved_path = graph.save("support_rules.pkl")
    print(f"\nGraph saved to: {saved_path}")
    
    # Visualize (if matplotlib is available)
    try:
        graph.visualize("support_rules_graph.png")
        print("Graph visualization saved to: support_rules_graph.png")
    except:
        pass


if __name__ == "__main__":
    example_usage()