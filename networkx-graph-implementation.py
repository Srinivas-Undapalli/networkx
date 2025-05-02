import networkx as nx
import json
import pickle
import os

class GraphDatabase:
    """
    A class to manage a property graph using NetworkX, with support for offline storage and loading.
    """
    
    def __init__(self, name="my_graph"):
        """
        Initialize a new graph database.
        
        Args:
            name (str): Name of the graph database
        """
        self.name = name
        # Using a directed graph (DiGraph) as it's more versatile
        # Use MultiDiGraph if you need multiple edges between the same nodes
        self.graph = nx.DiGraph(name=name)
    
    def add_node(self, node_id, **properties):
        """
        Add a node with properties to the graph.
        
        Args:
            node_id: Unique identifier for the node
            **properties: Key-value pairs of node properties
        """
        self.graph.add_node(node_id, **properties)
        return node_id
    
    def add_edge(self, source_id, target_id, edge_type=None, **properties):
        """
        Add an edge with properties between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type (str, optional): Type of relationship
            **properties: Key-value pairs of edge properties
        """
        if edge_type:
            properties['type'] = edge_type
            
        self.graph.add_edge(source_id, target_id, **properties)
        return (source_id, target_id)
    
    def get_node(self, node_id):
        """
        Get a node and its properties.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            dict: Node properties or None if node doesn't exist
        """
        if node_id in self.graph.nodes:
            return self.graph.nodes[node_id]
        return None
    
    def get_edge(self, source_id, target_id):
        """
        Get an edge and its properties.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            
        Returns:
            dict: Edge properties or None if edge doesn't exist
        """
        if self.graph.has_edge(source_id, target_id):
            return self.graph.edges[source_id, target_id]
        return None
    
    def update_node(self, node_id, **properties):
        """
        Update node properties.
        
        Args:
            node_id: ID of the node to update
            **properties: Properties to update
        """
        if node_id in self.graph.nodes:
            for key, value in properties.items():
                self.graph.nodes[node_id][key] = value
            return True
        return False
    
    def update_edge(self, source_id, target_id, **properties):
        """
        Update edge properties.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            **properties: Properties to update
        """
        if self.graph.has_edge(source_id, target_id):
            for key, value in properties.items():
                self.graph.edges[source_id, target_id][key] = value
            return True
        return False
    
    def query_nodes(self, **conditions):
        """
        Query nodes that match all the given conditions.
        
        Args:
            **conditions: Property conditions to match (exact match)
            
        Returns:
            list: List of node IDs that match the conditions
        """
        matching_nodes = []
        
        for node_id, attrs in self.graph.nodes(data=True):
            matches = True
            for key, value in conditions.items():
                if key not in attrs or attrs[key] != value:
                    matches = False
                    break
            if matches:
                matching_nodes.append(node_id)
                
        return matching_nodes
    
    def query_neighbors(self, node_id, edge_type=None, direction="out"):
        """
        Query neighbors of a node, optionally filtered by edge type.
        
        Args:
            node_id: ID of the node
            edge_type (str, optional): Filter by edge type
            direction (str): "out" for outgoing edges, "in" for incoming, "both" for both
            
        Returns:
            list: List of neighbor node IDs
        """
        neighbors = []
        
        if direction in ["out", "both"]:
            for _, neighbor in self.graph.out_edges(node_id):
                edge_attrs = self.graph.edges[node_id, neighbor]
                if edge_type is None or ('type' in edge_attrs and edge_attrs['type'] == edge_type):
                    neighbors.append(neighbor)
                    
        if direction in ["in", "both"]:
            for neighbor, _ in self.graph.in_edges(node_id):
                edge_attrs = self.graph.edges[neighbor, node_id]
                if edge_type is None or ('type' in edge_attrs and edge_attrs['type'] == edge_type):
                    neighbors.append(neighbor)
                    
        return neighbors
    
    def save_pickle(self, filepath=None):
        """
        Save the graph to a pickle file (binary).
        
        Args:
            filepath (str, optional): Path to save the file. If None, uses the graph name.
            
        Returns:
            str: Path to the saved file
        """
        if filepath is None:
            filepath = f"{self.name}.pkl"
            
        with open(filepath, 'wb') as f:
            pickle.dump(self.graph, f)
            
        return filepath
    
    def load_pickle(self, filepath=None):
        """
        Load the graph from a pickle file.
        
        Args:
            filepath (str, optional): Path to the file. If None, uses the graph name.
            
        Returns:
            bool: True if loading was successful
        """
        if filepath is None:
            filepath = f"{self.name}.pkl"
            
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.graph = pickle.load(f)
            return True
        return False
    
    def save_json(self, filepath=None):
        """
        Save the graph to a JSON file (human-readable).
        
        Args:
            filepath (str, optional): Path to save the file. If None, uses the graph name.
            
        Returns:
            str: Path to the saved file
        """
        if filepath is None:
            filepath = f"{self.name}.json"
            
        # Convert to node-link format
        data = nx.node_link_data(self.graph)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        return filepath
    
    def load_json(self, filepath=None):
        """
        Load the graph from a JSON file.
        
        Args:
            filepath (str, optional): Path to the file. If None, uses the graph name.
            
        Returns:
            bool: True if loading was successful
        """
        if filepath is None:
            filepath = f"{self.name}.json"
            
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.graph = nx.node_link_graph(data)
            return True
        return False
    
    def save_graphml(self, filepath=None):
        """
        Save the graph to a GraphML file (XML-based, interoperable).
        
        Args:
            filepath (str, optional): Path to save the file. If None, uses the graph name.
            
        Returns:
            str: Path to the saved file
        """
        if filepath is None:
            filepath = f"{self.name}.graphml"
            
        nx.write_graphml(self.graph, filepath)
        return filepath
    
    def load_graphml(self, filepath=None):
        """
        Load the graph from a GraphML file.
        
        Args:
            filepath (str, optional): Path to the file. If None, uses the graph name.
            
        Returns:
            bool: True if loading was successful
        """
        if filepath is None:
            filepath = f"{self.name}.graphml"
            
        if os.path.exists(filepath):
            self.graph = nx.read_graphml(filepath)
            return True
        return False
    
    def get_statistics(self):
        """
        Get basic statistics about the graph.
        
        Returns:
            dict: Graph statistics
        """
        return {
            'name': self.name,
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'is_directed': self.graph.is_directed(),
            'node_properties': self._get_property_stats(self.graph.nodes(data=True)),
            'edge_properties': self._get_property_stats(self.graph.edges(data=True))
        }
    
    def _get_property_stats(self, items):
        """Helper method to get statistics about node/edge properties"""
        property_counts = {}
        for _, data in items:
            for key in data:
                if key in property_counts:
                    property_counts[key] += 1
                else:
                    property_counts[key] = 1
        return property_counts


# Example usage
def example_usage():
    # Create a new graph database
    db = GraphDatabase("social_network")
    
    # Add nodes with properties
    db.add_node("user1", name="Alice", age=30, location="New York")
    db.add_node("user2", name="Bob", age=25, location="San Francisco")
    db.add_node("user3", name="Charlie", age=35, location="New York")
    db.add_node("post1", content="Hello world!", timestamp="2025-05-01")
    db.add_node("post2", content="Graph databases are awesome!", timestamp="2025-05-02")
    
    # Add edges with properties
    db.add_edge("user1", "user2", edge_type="FRIEND", since="2023-01-15")
    db.add_edge("user2", "user3", edge_type="FRIEND", since="2024-03-20")
    db.add_edge("user1", "post1", edge_type="CREATED", timestamp="2025-05-01T10:30:00")
    db.add_edge("user2", "post2", edge_type="CREATED", timestamp="2025-05-02T14:45:00")
    db.add_edge("user3", "post1", edge_type="LIKED", timestamp="2025-05-01T11:15:00")
    
    # Query examples
    print("Users in New York:", db.query_nodes(location="New York"))
    print("Bob's friends:", db.query_neighbors("user2", edge_type="FRIEND"))
    print("People who liked post1:", db.query_neighbors("post1", edge_type="LIKED", direction="in"))
    
    # Save the graph to different formats
    pkl_path = db.save_pickle("social_network.pkl")
    json_path = db.save_json("social_network.json")
    graphml_path = db.save_graphml("social_network.graphml")
    
    print(f"Graph saved to: {pkl_path}, {json_path}, and {graphml_path}")
    
    # Load a graph from a file
    new_db = GraphDatabase("loaded_graph")
    if new_db.load_json("social_network.json"):
        print("Graph loaded successfully!")
        print("Graph statistics:", new_db.get_statistics())


if __name__ == "__main__":
    example_usage()