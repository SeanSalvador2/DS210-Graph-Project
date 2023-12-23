use crate::file_and_graph_io::*;
use crate::components::*;
use crate::printing_functions::*;
use crate::distance_functions::*;

fn main() {
    //FIRST SECTION IS ALL SAMPLE DATA OUTPUTS CAN ACTUALLY BE PRINTED WITHOUT THOUSANDS OF LINES
    //small sample of real data example::
    println!("FIRST SECTION OF OUTPUT IS ALL SAMPLE DATA SO DATA IS DIGESTABLE:");
    let (mut sample_edges, sample_node_count, sample_node_set) = read_file(&String::from("citation_sample.txt"));

    //real data::
    //let (mut edges, node_count, node_set) = read_file(&String::from("citation_graph.txt"));
    println!("LIST OF EDGES:\nThe edges list for the data is: {:?} \n", sample_edges);
    println!("NODE COUNT:\nThe node_count for this data is {}", sample_node_count);

    //node_mapping is useful for converting from original labels to new labels
    //sorted_nodes is useful for converting from new labels to original labels
    let (sample_node_mapping,sample_sorted_nodes) = node_mapping(sample_node_set);
    println!("{:?}", sample_node_mapping);
    println!("The sorted nodes are {:?}", sample_sorted_nodes);
    
    //represented as a directed graph:
    println!("\nMaintaining the direction of the citations --> DIRECTED GRAPH ADJACENCY LIST:");
    let sample_dir_graph_list = make_adjacency_list(&mut sample_edges, sample_node_count, &sample_node_mapping,String::from("directed"));
    //Just to make sure the function is properly working, print the sample graph not the real data graph since that has thousands of nodes
    //First: print the sample directed adjacency list with the new labels
    println!("New labels:");
    print_adj_list(&sample_dir_graph_list, false, &sample_sorted_nodes);
    //Next: print the sample adjacency list with the original labels
    println!("Old labels:");
    print_adj_list(&sample_dir_graph_list, true, &sample_sorted_nodes);
    //example BFS
    let bfs_distance = bfs(&sample_dir_graph_list, 9, sample_node_count);
    //print the distances 
    print_bfs_distance(bfs_distance, sample_node_count);

    //represented as an undirected graph: 
    println!("\nIgnoring the direction of the citations --> UNDIRECTED GRAPH ADJCACENCY LIST ");
    let sample_undir_graph_list = make_adjacency_list(&mut sample_edges, sample_node_count, &sample_node_mapping, String::from("undirected"));
    //print the sample adjacency list with the new labels
    println!("New labels:");
    print_adj_list(&sample_undir_graph_list, false, &sample_sorted_nodes);
    println!("Old labels:");
    //print the adjacency list with the original labels
    print_adj_list(&sample_undir_graph_list, true, &sample_sorted_nodes);

    //calculate all distances using sample data (so output isn't too long)
    println!("\nDISTANCE CALCULATIONS FOR ALL NODES TO ALL NODES:");
    calculate_all_distances(&sample_undir_graph_list, sample_node_count);
    
    //calculate connected components
    let components_map = connected_components(&sample_undir_graph_list, sample_node_count); 

    //calculate how many nodes are in each component 
    for (component, nodes_in_component) in components_map.iter().enumerate() {
        println!("Component {} has {} nodes", component+1, nodes_in_component.len());
    }

    //observe that node 1 has 34001 nodes - which is basically all of them (only 34546 total)
    //Thus let's focus on component 1
    let main_component = &components_map[0];
    //println!("REAL DATA: Component 1 has nodes: {:?}", main_component); //don't print because there are thousands of nodes in this component


    //Calculate the average distance and the max distance between 100 randomly selected pairs of nodes in the main component 
    println!("\nDISTANCE STATISTICS:");
    let sample_size:usize = 1000;
    let (avg_dist, max_dist) = sample_avg_and_max_dist_in_component(main_component, &sample_undir_graph_list, sample_node_count, sample_size);
    println!("The average distance between the random nodes is {}\nThe max distance between the random nodes is {}", avg_dist, max_dist);




    //REAL DATA SECTION: 
    println!("\n\n\nTHE FOLLOWING IS THE RESULTS FROM MY FULL/REAL GRAPH DATA:");
    let (mut edges, node_count, node_set) = read_file(&String::from("citation_graph.txt"));
    println!("\nNODE COUNT:\nThe node_count for this data is {}", node_count);

    //node_mapping is useful for converting from original labels to new labels
    //sorted_nodes is useful for converting from new labels to original labels
    let (node_mapping, sorted_nodes) = node_mapping(node_set);

    //To allow for nodes to be better connected, I chose to use an undirected graph representation of my data:
    let undir_graph_list = make_adjacency_list(&mut edges, node_count, &node_mapping, String::from("undirected"));

    //calculate connected components
    let components_map = connected_components(&undir_graph_list, node_count); 

    //calculate how many nodes are in each component 
    for (component, nodes_in_component) in components_map.iter().enumerate() {
        println!("Component {} has {} nodes", component+1, nodes_in_component.len());
    }

    //observe that node 1 has 34001 nodes - which is basically all of them (only 34546 total)
    //Thus let's focus on component 1
    let main_component = &components_map[0];

    //Calculate the average distance and the max distance between 1000 randomly selected pairs of nodes in the main component 
    println!("\nDISTANCE STATISTICS:");
    let sample_size:usize = 500;
    let (avg_dist, max_dist) = sample_avg_and_max_dist_in_component(main_component, &undir_graph_list, node_count, sample_size);
    println!("The average distance between the random nodes is {}\nThe max distance between the random nodes is {}", avg_dist, max_dist);

    //Calculate node with the largest in degree centrality
    println!("\nIN DEGREE CENTRALITY STATISTICS:");
    let dir_graph_list = make_adjacency_list(&mut edges, node_count, &node_mapping, String::from("directed"));
    //takes about 30 seconds to run, is run in parallel 
    let (most_popular_node, highest_i_degree) = find_highest_in_degree_centrality(&dir_graph_list); 
    println!("Node {}, has {} incoming nodes", most_popular_node, highest_i_degree);
    println!("This node's original label was {}", sorted_nodes[most_popular_node]);
}

mod file_and_graph_io {
    use std::fs::File;
    use std::io::prelude::*;
    use std::collections::{HashSet,HashMap};
    pub fn read_file(path: &str) -> (Vec<(usize,usize)>, usize, HashSet<usize>) {
        //read in file so it gives one thing: a vector of tuples that represents every single connection in the file --> list of edges
            //also give node count 
        let mut edges: Vec<(usize, usize)> = Vec::new(); 
        let file = File::open(path).expect("Could not open file");
        let buf_reader = std::io::BufReader::new(file).lines();
        let mut node_set = HashSet::new();
        for line in buf_reader {
            let line_str = line.expect("Error reading");
            let v: Vec<&str> = line_str.trim().split('\t').collect(); // Split based on tabs --> values in file are tab separated
            //for each line, take the first and second numbers as usize numbers
            let x = v[0].parse::<usize>().unwrap();
            let y = v[1].parse::<usize>().unwrap();
            //dbg!(x, y);
            //push the nodes in as a tuple
            edges.push((x, y)); 

            //add to the node set (will only add if the set doesn't already have it)
            node_set.insert(x);
            node_set.insert(y);

            //observe that the starting node count is 32158 --> not all the destination nodes are starting nodes 
            //when also including the destination nodes --> 34546 
        }
        let node_count:usize = node_set.len();
        return (edges, node_count, node_set)
    }

    pub fn node_mapping(node_set:HashSet<usize>) -> (HashMap<usize,usize>, Vec<usize>) {
        //this function maps my noncontigous node labels to sequential/continous labels ranging from 0 to n-1 
        //returns the node_mapping (keys are original labels, values are new contigous labels) and the sorted_nodes 

        //Convert the node hashset into a vector so it can be sorted
            //into_iter makes the set iterable
            //collect collects the elements into the vector
        let mut sorted_nodes: Vec<usize> = node_set.into_iter().collect();
        sorted_nodes.sort();

        //now make the labels contigous ranging from 0 to n-1 
            //use a HashMap to keep track of the conversion so we can later map the original node ids to their new labels/indexes
        let mut node_to_index_mapping: HashMap<usize,usize> = HashMap::new();
        for (index, &node) in sorted_nodes.iter().enumerate() {
            //map original node ID to its index in the sorted order 
            node_to_index_mapping.insert(node,index);
        }
        return (node_to_index_mapping, sorted_nodes);
    }

    pub fn make_adjacency_list(edges: &mut Vec<(usize,usize)>, node_count:usize, node_mapping:&HashMap<usize,usize>, graph_type:String) -> Vec<Vec<usize>> { 
        //this function uses the list of edges and the node_mapping to create the adjacency list representation for a directed graph
        //first convert the edges list to the new contigous labels

        //iterate through the list of edges, use the node mapping to convert the nodes accordingly 
        let mut new_edges:Vec<(usize,usize)> = Vec::new();
        for (start_node, end_node) in edges.iter() {
            let new_start_node = *node_mapping.get(start_node).unwrap(); //use unwrap because .get() returns an option type on HashMaps
            let new_end_node = *node_mapping.get(end_node).unwrap();
            new_edges.push((new_start_node,new_end_node))
        }
        //now create the adjacency list for the directed graph
        let mut graph_list:Vec<Vec<usize>> = vec![vec![];node_count];
        for (u, v) in new_edges.iter() {
            if graph_type == String::from("directed") {
                graph_list[*u].push(*v); //only push one way for directed graph
            }
            else if graph_type == String::from("undirected") {
                graph_list[*u].push(*v);
                graph_list[*v].push(*u); //push both ways for undirected graph
            }
        }
        return graph_list;
    }
}
mod components {
    pub fn connected_components(adj_list: &Vec<Vec<usize>>, node_count: usize) -> Vec<Vec<usize>> {
        let mut component: Vec<Option<usize>> = vec![None;node_count];
        let mut component_count = 0;
        //Use a HashMap to keep track of what nodes are in each component
        let mut components_list: Vec<Vec<usize>> = Vec::new();
        for v in 0..node_count {
            if let None = component[v] {
                component_count += 1;
                let nodes_in_comp = mark_component_bfs(v, &adj_list, &mut component, component_count);
                components_list.push(nodes_in_comp)
        }
        }
        //print the components
        println!("\nCOMPONENT COUNT: \nThere are {component_count} components in this graph");
        return components_list
    }
    fn mark_component_bfs(vertex:usize, adj_list:&Vec<Vec<usize>>, component:&mut Vec<Option<usize>>, component_no:usize) ->Vec<usize> {
        //make a vector to store nodes in the current component
        let mut nodes_in_component = Vec::new();
        component[vertex] = Some(component_no);
        
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(vertex);
        
        while let Some(v) = queue.pop_front() {
            nodes_in_component.push(v);
            //add the current to the list of nodes in the current component
            for w in adj_list[v].iter() {
                if let None = component[*w] {
                    component[*w] = Some(component_no);
                    queue.push_back(*w);
                }
            }
        }
        return nodes_in_component;
    }
}

mod printing_functions{
    pub fn print_adj_list(adj_list: &Vec<Vec<usize>>, original_labels: bool, sorted_nodes: &Vec<usize>) {
        //prints the adjacency list with the original node labels or the new labels
            //True --> yes original labels, False --> new labels ranging from 0 to n-1
        if original_labels == false {
            for (i, l) in adj_list.iter().enumerate() {
                if !l.is_empty() { //only print the nodes that have any outgoing edges
                    println!("{}, {:?}", i, *l);
                }
            }
        }
        else if original_labels == true {
            for (i, l) in adj_list.iter().enumerate() {
                let start_node = sorted_nodes[i];
                let mut end_nodes: Vec<usize> = Vec::new();
                //use the sorted_nodes vector to map the new labels back to their original labels
                    //the node with label 0 is the 0th element in the sorted_nodes vector 
                for j in l {
                    //go through the destination node vector, convert each element back to the original label and push it to end_nodes vec
                    let original_end_node_label = sorted_nodes[*j];
                    end_nodes.push(original_end_node_label)
                }
                if !end_nodes.is_empty() { //only print the nodes that have any outgoing edges
                    println!("{}, {:?}", start_node, end_nodes);
                }
            }
        }
    
    }
    
    pub fn print_bfs_distance(distance: Vec<Option<u32>>, node_count: usize) {
        print!("vertex:distance -->");
        for v in 0..node_count {
            match distance[v] {
                Some(d) => print!("   {}:{}", v, d),
                None => print!("   {}:INF", v), //INF means the node is not reachable --> infinite distance between them 
            }
        }
        println!("");
    }
    
}
mod distance_functions {
    use rand::Rng; 
    use std::collections::VecDeque;
    use crate::printing_functions::print_bfs_distance;
    pub fn bfs(adj_list: &Vec<Vec<usize>>, start:usize, node_count:usize) -> Vec<Option<u32>> {
        let mut distance: Vec<Option<u32>> = vec![None;node_count];
        distance[start] = Some(0); //we know the distance of the starting node to itself is 0
        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(start);
    
        while let Some(v) = queue.pop_front() { //look at the new unprocessed vertex
            for u in adj_list[v].iter() {
                //look at all the unprocessed neighbors of v
                if distance[*u].is_none() {
                    //if distance to u hasn't been found yet, calculate it and update the distance vector
                    if let Some(dist_v) = distance[v] {
                        distance[*u] = Some(dist_v + 1);
                        queue.push_back(*u); //add node u to the back of the queue
                    } else {
                        // Handle the case where distance[v] is None
                        println!("Error: distance[{}] is None!", v);
                        continue 
                    }
                }
            }
        }
        
        return distance
    }
    pub fn calculate_all_distances(adj_list: &Vec<Vec<usize>>, node_count:usize) -> Vec<Vec<Option<u32>>> {
        let mut all_distances = Vec::new();
        for i in 0..node_count {
            println!("Distances from node {}", i);
            let distance = bfs(adj_list, i, node_count);
            all_distances.push(distance.clone());
            print_bfs_distance(distance, node_count);
        }
        return all_distances;
    }
    pub fn sample_avg_and_max_dist_in_component(main_component: &Vec<usize>, adj_list: &Vec<Vec<usize>>, node_count: usize, sample_size:usize)-> (f32,usize) {
        //This function calculates the average distance between n random pairs of nodes in a specific component/subset of connected nodes
            //focusing on a specific components because distances between disconnected 
        let mut distances:Vec<usize> = Vec::new();
        let mut max_distance = 0;
        for _ in 0..sample_size{
            let distance = _distance_between_2_random_nodes(adj_list, main_component, node_count);
            distances.push(distance);
            if distance > max_distance {
                max_distance = distance;
            }
        }
        let sum_distances:usize = distances.iter().sum();
        let average_distance = sum_distances as f32/ distances.len() as f32;
        return (average_distance, max_distance);
    }
    fn _distance_between_2_random_nodes(adj_list: &Vec<Vec<usize>>, main_component: &Vec<usize>,node_count:usize) -> usize {
        //this function finds the distance between two randomly selected nodes from a specific component/set of nodes
        let component_len = main_component.len();
        let rand_index_1 = rand::thread_rng().gen_range(0..component_len); //randomly select integer from 0 to number of elements in component
        let rand_index_2 = rand::thread_rng().gen_range(0..component_len);
        let rand_1 = main_component[rand_index_1];
        let rand_2 = main_component[rand_index_2];

        //use bfs function to see distance from rand_1 node to rand_2 node
        let distances:Vec<Option<u32>> = bfs(&adj_list, rand_1, node_count);
        let mut node_distance:usize = 1000000;
        //get the distance to the rand_2 node --> must use match or if let to extract from Option type
        if let Some(distance_to_rand_2) = &distances[rand_2] {
            node_distance = *distance_to_rand_2 as usize;
        }
        return node_distance
    }
    fn _calculate_in_degree_centrality(adj_list:&Vec<Vec<usize>>, starting_node:usize) -> usize {
        //this functions calculates the degree centrality of a node or the number of edges that connect/point to a node
            //assumes directed graph  
        let mut incoming_edges_count:usize = 0;  //counter for keeping track of incoming edges
        //iterate over each node in the adjacency list
        for neighbors in adj_list.iter() {
            if neighbors.contains(&starting_node) {
                incoming_edges_count += 1;
            }
        }
        return incoming_edges_count
    }
    use rayon::prelude::*;
    pub fn find_highest_in_degree_centrality(adj_list: &Vec<Vec<usize>>) -> (usize, usize) {
        //this function finds the node with the highest in degree centrality 
        // Create a mutable reference to a vector to store node and in-degree pairs
        let node_and_in_degrees: Vec<(usize, usize)> = (0..adj_list.len())
            .into_par_iter()  // Use par_iter() for parallel iteration
            .map(|node| {
                //for each node, calculate its degree in parallel
                let in_degree: usize = _calculate_in_degree_centrality(adj_list, node);
                (node, in_degree) //return a tuple 
            })
            .collect();
    
        // Find the node with the maximum in-degree
        let mut most_pop_node: usize = 0;
        let mut max_in_degree: usize = 0;
    
        for (node, in_degree) in node_and_in_degrees {
            if in_degree > max_in_degree {
                max_in_degree = in_degree;
                most_pop_node = node;
            }
        }
    
        // Return the result as a tuple
        return (most_pop_node, max_in_degree);
    }


}
#[cfg(test)]
mod tests {
    use crate::file_and_graph_io::*;
    use crate::connected_components;
    use crate::distance_functions::*;
    #[test]
    fn test_read_file_edges_list() {
        //use smaller toy data to test function
        let (edges, node_count, _) = read_file(&String::from("citation_toy.txt"));
        //check if list of edges was read properly and that node count was counted properly 
        assert_eq!(edges, vec![(0, 1), (0, 2), (1, 2), (1, 9), (2, 3), (2, 4), (4, 3), (4, 5), (4, 6), (5, 6), (6, 7), (6, 8), (8, 7)]);
        assert_eq!(node_count,10);
    }
    #[test]
    fn test_node_mapping_sample_data() {
        //test using the first 10 lines of my real data so its digestable
        use std::collections::HashMap;
        let (_edges, _node_count, node_set) = read_file(&String::from("citation_sample.txt"));
        let (node_mapping,sorted_nodes) = node_mapping(node_set);
        let node_pairs = vec![(9702314, 10),(9907233, 12),(9502274, 5),(9501357, 4),(9504304, 7),(9704296, 11),(9302247, 2),(9301253, 1),
                                        (9301206, 0),(9311274, 3),(9505235, 8),(9502335, 6),(9606402, 9)];
        let node_hashmap: HashMap<_, _> = node_pairs.into_iter().collect();
        //check if node mapping is right, then check if nodes sorted correctly
        assert_eq!(node_mapping,node_hashmap); 
        assert_eq!(sorted_nodes,vec![9301206, 9301253, 9302247, 9311274, 9501357, 9502274, 9502335, 9504304, 9505235, 9606402, 9702314, 9704296, 9907233]);
    }
    #[test]
    fn test_make_adj_list_directed_graph_sample_data() {
        let (mut edges, node_count, node_set) = read_file(&String::from("citation_sample.txt"));
        let (node_mapping,_) = node_mapping(node_set);
        let dir_graph_list = make_adjacency_list(&mut edges, node_count, &node_mapping,String::from("directed"));
        let expected_adj_list:Vec<Vec<usize>> = vec![vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![0, 5], vec![2, 4], vec![3, 6, 9], vec![1, 7, 8]];
        assert_eq!(dir_graph_list, expected_adj_list);
    }
    #[test]
    fn test_make_adj_list_undirected_graph_sample_data() {
        let (mut edges, node_count, node_set) = read_file(&String::from("citation_sample.txt"));
        let (node_mapping,_) = node_mapping(node_set);
        let undir_graph_list = make_adjacency_list(&mut edges, node_count, &node_mapping,String::from("undirected"));
        let expected_adj_list:Vec<Vec<usize>> =  vec![vec![9], vec![12], vec![10], vec![11], vec![10], vec![9], vec![11], vec![12], vec![12], vec![11, 0, 5], vec![2, 4], vec![3, 6, 9], vec![1, 7, 8]];
        assert_eq!(undir_graph_list, expected_adj_list);
    }
    #[test]
    fn test_connected_components_sample_data() {
        let (mut edges, node_count, node_set) = read_file(&String::from("citation_sample.txt"));
        let (node_mapping,_) = node_mapping(node_set);
        let undir_graph_list = make_adjacency_list(&mut edges, node_count, &node_mapping,String::from("undirected"));
        let components_map = connected_components(&undir_graph_list, node_count); 
        assert_eq!(components_map,vec![vec![0, 9, 11, 5, 3, 6], vec![1, 12, 7, 8], vec![2, 10, 4]] );
    }
    #[test]
    fn test_bfs_sample_data() {
        //test BFS, using a starting point of node 9
        let (mut edges, node_count, node_set) = read_file(&String::from("citation_sample.txt"));
        let (node_mapping,_) = node_mapping(node_set);
        let undir_graph_list = make_adjacency_list(&mut edges, node_count, &node_mapping,String::from("undirected"));
        let undir_distance = bfs(&undir_graph_list, 0, node_count);
        assert_eq!(undir_distance,vec![Some(0), None, None, Some(3), None, Some(2), Some(3), None, None, Some(1), None, Some(2), None]);
    }
    #[test]
    fn test_avg_max_distance_from_random_sample() {
        //test my average/max distance function --> since it uses random pairs, must use a tolerance as an acceptable answer (set lower and upper bound)
        //use the sample data --> only a small number of nodes/edges so average and max distance is predictable 
        let (mut edges, node_count, node_set) = read_file(&String::from("citation_sample.txt"));
        let (node_mapping,_) = node_mapping(node_set);
        let undir_graph_list = make_adjacency_list(&mut edges, node_count, &node_mapping,String::from("undirected"));
        let components_map = connected_components(&undir_graph_list, node_count); 
        let main_component = &components_map[0];
        let sample_size:usize = 1000;
        let (avg_dist, max_dist) = sample_avg_and_max_dist_in_component(main_component, &undir_graph_list, node_count, sample_size);
        assert!(avg_dist>1.0 && avg_dist < 2.0);
        assert!(max_dist > 1 && max_dist<= 3);
    }
    

}

    //file comments: 
    //# Directed graph (each unordered pair of nodes is saved once): Cit-HepPh.txt 
    //# Paper citation network of Arxiv High Energy Physics category
    //# Nodes: 34546 Edges: 421578
    //# FromNodeId	ToNodeId
