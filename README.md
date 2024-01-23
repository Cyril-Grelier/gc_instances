# Instances for Graph Coloring Problem and Weighted Vertex Coloring Problem

This repo lists several instances used in the Graph Coloring Problem (GCP) and the Weighted Vertex Coloring Problem (WVCP), original graph and reduced version.

## Usage

### add this module to your project :

    git submodule add https://github.com/Cyril-Grelier/gc_instances.git instances

### remove the module from your project :

    git config -f .git/config --remove-section submodule.instances
    git config -f .gitmodules --remove-section submodule.instances
    git add .gitmodules
    git rm --cached instances
    git add .gitmodules
    rm -rf instances
    rm -rf .git/modules/instances

### Update the module :

    cd instances/
    git pull origin main
    cd ..
    git add instances/
    git commit -m "submodule instance updated"
    git push

### Recompute reduction (python 3.9+)

    ./build.sh
    source venv/bin/activate
    # you might need to edit main function in reduction.py before
    python reduction.py

### Check a solution from a reduced graph :

    python check_solution.py

## Organization of files

All instances can be found in `original_graphs`. The reduced version of the graphs can be found in `reduced_gcp` or `reduced_wvcp` (if the original graph has a weight file). You can find a summary of the reduction in `summary_reduction_PROBLEM.csv`.

The vertices of each instance of `reduced_PROBLEM` are sorted by weight then by degree if equal weight then by the sum of the weights of theirs neighbors if same degree.

You can find three types of files:

    - .col files : from the DIMACS challenge (vertex numbers start at 1)
    - .edgelist files : each line is an edge between two nodes (vertex numbers start at 0)
    - .col.w files : line 1 is the weight of vertex 0, line 2 the weight of vertex 1, ...

The original files may have been modified to remove lines (from wcol format) or removing empty comments, but all the graphs remain the same (number of nodes and edges, weights,...).

If you find an error please inform us.

## About the reduction

Details of the reduction are published in this article : https://www.ijcai.org/proceedings/2023/214
You can find and exemple of the reduction here : https://cyril-grelier.github.io/graph_coloring/2023/11/10/demo_reduction.html

Based reduction from :

- P. C. Cheeseman, B. Kanefsky, and W. M. Taylor, “Where the Really Hard Problems Are,” presented at the IJCAI, Jan. 1991. Available: https://openreview.net/forum?id=SkZHONzdbB

- Y. Wang, S. Cai, S. Pan, X. Li, and M. Yin, “Reduction and Local Search for Weighted Graph Coloring Problem,” AAAI, vol. 34, no. 03, Art. no. 03, Apr. 2020, doi: 10.1609/aaai.v34i03.5624.

## Best scores

`best_score_wvcp.txt` and `best_score_gcp.txt` list the best known scores for each problem.

On each line you can find : the name of the instance, the score, a `*` if the score is proved optimal otherwise a `-` if non proved optimal and `?` if the score is unknown.

Most of these scores are reported scores, the solutions linked to the score are not always available to validate the score and the time spent to reach these scores depends on the article.

See those article for the origin of the scores :

- L. Moalic and A. Gondran, “Variations on memetic algorithms for graph coloring problems,” J Heuristics, vol. 24, no. 1, pp. 1–24, Feb. 2018, doi: 10.1007/s10732-017-9354-9.

- P. Galinier, A. Hertz, and N. Zufferey, “An adaptive memory algorithm for the k-coloring problem,” Discrete Applied Mathematics, vol. 156, no. 2, pp. 267–279, Jan. 2008, doi: 10.1016/j.dam.2006.07.017.

- Y. Wang, S. Cai, S. Pan, X. Li, and M. Yin, “Reduction and Local Search for Weighted Graph Coloring Problem,” AAAI, vol. 34, no. 03, Art. no. 03, Apr. 2020, doi: 10.1609/aaai.v34i03.5624.

- Nogueira Bruno, Eduardo Tavares, et Paulo Maciel. «Iterated Local Search with Tabu Search for the Weighted Vertex Coloring Problem». Computers & Operations Research 125 (1 janvier 2021): 105087. https://doi.org/10.1016/j.cor.2020.105087.

- Goudet Olivier, Cyril Grelier, and Jin-Kao Hao. “A Deep Learning Guided Memetic Framework for Graph Coloring Problems.” Knowledge-Based Systems 258 (December 22, 2022): 109986. https://doi.org/10.1016/j.knosys.2022.109986.

- Cyril Grelier, Olivier Goudet, Jin-Kao Hao. On monte carlo tree search for weighted vertex voloring. Proceedings of the 22nd European Conference on Evolutionary Computation in Combinatorial Optimization (EvoCOP 2022), 20-22 April, 2022, Lecture Notes in Computer Science 13222, pages 1-16. https://link.springer.com/chapter/10.1007/978-3-031-04148-8_1 - https://github.com/Cyril-Grelier/gc_wvcp_mcts

## List of instances

DIMAC_large.txt, DIMAC_small.txt, pxx.txt and rxx.txt are the lists of instances used in the state of the art (for WVCP) to compare the algorithms. other.txt lists instances in the directory but not currently used for comparison in the state of the art (for WVCP).

## source of the files :

- various ones for GCP : https://sites.google.com/site/graphcoloring/files
- wvcp_original: http://www.info.univ-angers.fr/pub/hao/wvcp.html
- graph_coloring : DIMACS Challenge II
- bandwidth_multicoloring_instances : https://mat.gsia.cmu.edu/COLOR04/
- shared by other teams working on WVCP
