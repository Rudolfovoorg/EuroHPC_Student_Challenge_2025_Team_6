// dsatur_pair_chiga_parallel.cpp
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <string>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>

using namespace std;
using namespace std::chrono;

const int INF = 1000000000;

// Variabili globali per tenere traccia del tempo e della soluzione migliore
steady_clock::time_point startTime;
bool searchCompleted = true;   // diventa false se si supera il time limit
int bestSolution = INF;        // numero minimo di colori trovato finora
vector<int> bestColoring;      // assegnamento migliore (di lunghezza n, per i vertici originali)

// Variabili MPI globali
int mpi_rank, mpi_size;

// ----------------------------------------------------------------------
// Struttura del grafo
// ----------------------------------------------------------------------
struct Graph {
    int n;                        // numero totale di vertici
    vector<vector<bool>> adj;     // matrice di adiacenza (dimensione n x n)

    Graph(int n_) : n(n_) {
        adj.assign(n, vector<bool>(n, false));
    }
    Graph() : n(0) {}
    
    // -------------------------------
    // Heuristica per il massimo clique (greedy)
    // Restituisce un pair: (dimensione clique, vettore dei vertici della clique)
    // -------------------------------
    pair<int, vector<int>> heuristicMaxClique() const {
        vector<pair<int,int>> degVertex;
        for (int i = 0; i < n; i++) {
            int deg = 0;
            for (int j = 0; j < n; j++)
                if (adj[i][j])
                    deg++;
            degVertex.push_back({deg, i});
        }
        sort(degVertex.begin(), degVertex.end(), greater<pair<int,int>>());
        vector<int> clique;
        for (auto &p : degVertex) {
            int v = p.second;
            bool canAdd = true;
            for (int u : clique)
                if (!adj[v][u]) { canAdd = false; break; }
            if (canAdd)
                clique.push_back(v);
        }
        // In C++11 la sintassi "return { ... }" è ammessa.
        return { static_cast<int>(clique.size()), clique };
    }
    
    // -------------------------------
    // Heuristica per il coloring greedy (DSATURh)
    // Restituisce un pair: (numero di colori usati, vettore di colorazione)
    // -------------------------------
    pair<int, vector<int>> heuristicColoring() const {
        vector<int> color(n, -1);
        for (int v = 0; v < n; v++) {
            vector<bool> used(n, false);
            for (int u = 0; u < n; u++) {
                if (adj[v][u] && color[u] != -1)
                    used[color[u]] = true;
            }
            int c;
            for (c = 0; c < n; c++)
                if (!used[c])
                    break;
            color[v] = c;
        }
        int numColors = *max_element(color.begin(), color.end()) + 1;
        return { numColors, color };
    }
};

// ----------------------------------------------------------------------
// Lettura del grafo da file in formato COL
// ----------------------------------------------------------------------
Graph readGraphFromCOLFile(const string &filename) {
    ifstream infile(filename);
    if (!infile) {
        cerr << "Errore nell'apertura del file " << filename << endl;
        exit(1);
    }
    
    int n = 0, m = 0;
    string line;
    vector<pair<int,int>> edges;
    
    while(getline(infile, line)) {
        if (line.empty()) continue;
        if (line[0] == 'c') continue; // commento
        if (line[0] == 'p') {
            istringstream iss(line);
            string tmp;
            iss >> tmp >> tmp >> n >> m;
        }
        if (line[0] == 'e') {
            istringstream iss(line);
            char ch;
            int u, v;
            iss >> ch >> u >> v;
            // conversione da 1-indexed a 0-indexed
            edges.push_back({u - 1, v - 1});
        }
    }
    
    Graph g(n);
    for (auto &edge : edges) {
        int u = edge.first, v = edge.second;
        if(u >= 0 && u < n && v >= 0 && v < n)
            g.adj[u][v] = g.adj[v][u] = true;
    }
    
    return g;
}

// ----------------------------------------------------------------------
// Funzioni ausiliarie per la colorazione parziale
// ----------------------------------------------------------------------

// Restituisce gli indici dei vertici non ancora colorati
vector<int> getUncoloredVertices(const vector<int> &colors) {
    vector<int> uncolored;
    for (int i = 0; i < (int)colors.size(); i++) {
        if (colors[i] == -1)
            uncolored.push_back(i);
    }
    return uncolored;
}

// Dato un grafo g e un vettore di indici "vertices", restituisce il sottografo indotto
Graph getInducedSubgraph(const Graph &g, const vector<int>& vertices) {
    int n = vertices.size();
    Graph subG(n);
    for (int i = 0; i < n; i++){
        for (int j = i+1; j < n; j++){
            subG.adj[i][j] = subG.adj[j][i] = g.adj[vertices[i]][vertices[j]];
        }
    }
    return subG;
}

// Calcola i colori “feasible” per il vertice v (colori non usati dai suoi adiacenti)
// Considera i colori già usati (0...currentColors-1) e aggiunge anche la possibilità di usare un nuovo colore
vector<int> feasibleColorsForVertex(int v, const Graph &g, const vector<int> &colors, int currentColors) {
    vector<bool> used(currentColors, false);
    for (int u = 0; u < g.n; u++){
         if(g.adj[v][u] && colors[u] != -1) {
             int col = colors[u];
             if(col < currentColors)
                used[col] = true;
         }
    }
    vector<int> feas;
    for (int c = 0; c < currentColors; c++){
         if(!used[c])
             feas.push_back(c);
    }
    feas.push_back(currentColors); // il nuovo colore è sempre disponibile
    return feas;
}

// ----------------------------------------------------------------------
// Funzioni per il lower bound ausiliario (χGA)
// ----------------------------------------------------------------------

// Struttura per rappresentare un arco in un orientamento aciclico
struct Arc {
    int u, v;
};

// Data un grafo H, costruisce un orientamento aciclico del complemento di H
vector<Arc> buildAcyclicOrientationArcs(const Graph &H) {
    int n = H.n;
    vector<vector<bool>> comp(n, vector<bool>(n, false));
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            if(i == j) continue;
            comp[i][j] = !H.adj[i][j];
        }
    }
    vector<Arc> arcs;
    for (int i = 0; i < n; i++){
        for (int j = i+1; j < n; j++){
            if(comp[i][j]) {
                arcs.push_back({i, j});
            }
        }
    }
    return arcs;
}

// Costruisce il line–graph GA a partire dall’insieme di archi (ogni arco diventa un vertice)
vector<vector<int>> buildLineGraph(const vector<Arc> &arcs) {
    int m = arcs.size();
    vector<vector<int>> gaAdj(m);
    for (int i = 0; i < m; i++){
        for (int j = i+1; j < m; j++){
            // due archi sono adiacenti se condividono un estremo
            if(arcs[i].u == arcs[j].u || arcs[i].u == arcs[j].v ||
               arcs[i].v == arcs[j].u || arcs[i].v == arcs[j].v) {
                gaAdj[i].push_back(j);
                gaAdj[j].push_back(i);
            }
        }
    }
    return gaAdj;
}

// Algoritmo greedy per stimare la dimensione di un maximum stable set su GA
int heuristicMaxStableSet(const vector<vector<int>> &gaAdj) {
    int m = gaAdj.size();
    vector<bool> removed(m, false);
    int stableSize = 0;
    vector<int> order(m);
    for (int i = 0; i < m; i++) order[i] = i;
    sort(order.begin(), order.end(), [&](int a, int b) {
        return gaAdj[a].size() < gaAdj[b].size();
    });
    for (int v : order) {
        if (!removed[v]) {
            stableSize++;
            removed[v] = true;
            for (int u : gaAdj[v]) {
                removed[u] = true;
            }
        }
    }
    return stableSize;
}

// Calcola il lower bound ausiliario basato su χGA:
// LB_aux = currentColors + (|U| - α(GA)), dove U è l'insieme dei vertici non colorati
int computeAuxiliaryLowerBound(const Graph &g, const vector<int> &colors, int currentColors) {
    vector<int> uncolored = getUncoloredVertices(colors);
    int U = uncolored.size();
    if(U == 0) return currentColors;
    Graph H = getInducedSubgraph(g, uncolored);
    vector<Arc> arcs = buildAcyclicOrientationArcs(H);
    vector<vector<int>> gaAdj = buildLineGraph(arcs);
    int alpha = heuristicMaxStableSet(gaAdj);
    int auxBound = currentColors + (U - alpha);
    return auxBound;
}

// ----------------------------------------------------------------------
// Funzione φ: decide se calcolare il lower bound ausiliario
// (qui: se il numero di vertici colorati è tra il 18% e il 35% e se (bestSolution - currentColors) <= 10)
bool phiFunction(int coloredCount, int totalVertices, int currentColors, int bestSolution) {
    if(coloredCount >= 0.18 * totalVertices && coloredCount <= 0.35 * totalVertices && (bestSolution - currentColors) <= 10)
         return true;
    return false;
}

// ----------------------------------------------------------------------
// Funzione ricorsiva DSATUR–χGA parallelizzata
//
// La funzione ora accetta un ulteriore parametro "depth".
// Al livello 0 il ciclo sui rami (branching) viene partizionato staticamente tra i processi MPI.
void dsaturPairRecursive(const Graph &g, vector<int> &colors, int currentColors, double timeLimit, int depth) {
    double elapsed = duration_cast<duration<double>>(steady_clock::now() - startTime).count();
    if(elapsed >= timeLimit) {
        searchCompleted = false;
        return;
    }
    
    int n = g.n;
    int coloredCount = 0;
    for (int i = 0; i < n; i++)
        if (colors[i] != -1)
            coloredCount++;
    
    // Caso base: tutti i vertici sono colorati
    if(coloredCount == n) {
        if(currentColors < bestSolution) {
            bestSolution = currentColors;
            bestColoring = colors;
            cout << "Processo " << mpi_rank << ": nuova soluzione migliore trovata con " << bestSolution << " colori.\n";
        }
        return;
    }
    
    // Lower bound classico tramite clique euristica sul sottografo dei vertici non colorati
    vector<int> U = getUncoloredVertices(colors);
    Graph subG = getInducedSubgraph(g, U);
    auto cliquePair = subG.heuristicMaxClique();
    int cliqueSize = cliquePair.first;
    int lb = currentColors + cliqueSize;
    if(lb >= bestSolution) return;
    
    // Se il nodo è promettente, calcola il lower bound ausiliario
    if(phiFunction(coloredCount, n, currentColors, bestSolution)) {
        int auxLB = computeAuxiliaryLowerBound(g, colors, currentColors);
        if(auxLB >= bestSolution) return;
    }
    
    // Seleziona una coppia di vertici non colorati non adiacenti
    int v1 = -1, v2 = -1;
    bool foundPair = false;
    for (int i = 0; i < n && !foundPair; i++) {
        if(colors[i] == -1) {
            for (int j = i+1; j < n && !foundPair; j++) {
                if(colors[j] == -1 && !g.adj[i][j]) {
                    v1 = i;
                    v2 = j;
                    foundPair = true;
                }
            }
        }
    }
    // Se non esiste una coppia, i vertici non colorati formano una clique:
    if(!foundPair) {
        int needed = n - coloredCount;
        int totalColors = currentColors + needed;
        if(totalColors < bestSolution) {
            bestSolution = totalColors;
            vector<int> completeColoring = colors;
            int nextColor = currentColors;
            for (int i = 0; i < n; i++) {
                if(completeColoring[i] == -1) {
                    completeColoring[i] = nextColor;
                    nextColor++;
                }
            }
            bestColoring = completeColoring;
            cout << "Processo " << mpi_rank << ": nodo completo, bestSolution aggiornato a " << bestSolution << "\n";
        }
        return;
    }
    
    // --- Branch 1: forzo v1 e v2 ad avere lo stesso colore ---
    vector<int> feas1 = feasibleColorsForVertex(v1, g, colors, currentColors);
    vector<int> feas2 = feasibleColorsForVertex(v2, g, colors, currentColors);
    vector<int> common;
    for (int c : feas1) {
         if(find(feas2.begin(), feas2.end(), c) != feas2.end())
             common.push_back(c);
    }
    if(depth == 0) {
        int branchCounter = 0;
        for (int c : common) {
             if(branchCounter % mpi_size != mpi_rank) { branchCounter++; continue; }
             int newCurrentColors = currentColors;
             if(c == currentColors)
                  newCurrentColors = currentColors + 1;
             int origV1 = colors[v1], origV2 = colors[v2];
             colors[v1] = c;
             colors[v2] = c;
             dsaturPairRecursive(g, colors, newCurrentColors, timeLimit, depth+1);
             colors[v1] = origV1;
             colors[v2] = origV2;
             branchCounter++;
             if(!searchCompleted) return;
        }
    } else {
        for (int c : common) {
             int newCurrentColors = currentColors;
             if(c == currentColors)
                  newCurrentColors = currentColors + 1;
             int origV1 = colors[v1], origV2 = colors[v2];
             colors[v1] = c;
             colors[v2] = c;
             dsaturPairRecursive(g, colors, newCurrentColors, timeLimit, depth+1);
             colors[v1] = origV1;
             colors[v2] = origV2;
             if(!searchCompleted) return;
        }
    }
    
    // --- Branch 2: forzo v1 e v2 ad avere colori differenti ---
    if(depth == 0) {
        int branchCounter = 0;
        for (int c1 : feas1) {
             for (int c2 : feas2) {
                  if(c1 == c2) continue;
                  if(branchCounter % mpi_size != mpi_rank) { branchCounter++; continue; }
                  int newCurrentColors = currentColors;
                  if(c1 == currentColors || c2 == currentColors)
                      newCurrentColors = currentColors + 1;
                  int origV1 = colors[v1], origV2 = colors[v2];
                  colors[v1] = c1;
                  colors[v2] = c2;
                  dsaturPairRecursive(g, colors, newCurrentColors, timeLimit, depth+1);
                  colors[v1] = origV1;
                  colors[v2] = origV2;
                  branchCounter++;
                  if(!searchCompleted) return;
             }
        }
    } else {
        for (int c1 : feas1) {
             for (int c2 : feas2) {
                  if(c1 == c2) continue;
                  int newCurrentColors = currentColors;
                  if(c1 == currentColors || c2 == currentColors)
                      newCurrentColors = currentColors + 1;
                  int origV1 = colors[v1], origV2 = colors[v2];
                  colors[v1] = c1;
                  colors[v2] = c2;
                  dsaturPairRecursive(g, colors, newCurrentColors, timeLimit, depth+1);
                  colors[v1] = origV1;
                  colors[v2] = origV2;
                  if(!searchCompleted) return;
             }
        }
    }
}

// ----------------------------------------------------------------------
// main()
// ----------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if(argc < 3) {
        if(mpi_rank == 0)
            cerr << "Utilizzo: " << argv[0] << " <input_file> <time_limit_sec>\n";
        MPI_Finalize();
        return 1;
    }
    string inputFile = argv[1];
    double timeLimit = atof(argv[2]);  // ad es. 10.0 secondi

    // Solo il processo master legge il grafo
    Graph root = readGraphFromCOLFile(inputFile);
    int n = root.n;
    
    // Tutti i processi ricevono il grafo (semplice copia: in un vero ambiente si potrebbe distribuire in modo più efficiente)
    // In questo esempio assumiamo che il grafo sia piccolo e venga replicato in ogni processo.
    
    // Inizializza la colorazione parziale: tutti i vertici non sono colorati (valore -1)
    vector<int> colors(n, -1);
    int currentColors = 0;
    
    // Calcola un UB iniziale con una colorazione greedy (DSATURh)
    auto coloringPair = root.heuristicColoring();
    bestSolution = coloringPair.first;
    bestColoring = coloringPair.second;
    if(mpi_rank == 0)
        cout << "Upper bound iniziale (DSATURh): " << bestSolution << "\n";
    
    startTime = steady_clock::now();
    
    // Avvia la ricerca ricorsiva con depth = 0 (la suddivisione dei rami avviene solo a questo livello)
    dsaturPairRecursive(root, colors, currentColors, timeLimit, 0);
    
    // Riduzione per ottenere il miglior bestSolution globale
    int globalBest;
    MPI_Reduce(&bestSolution, &globalBest, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    // Diffondi il globalBest a tutti i processi
    MPI_Bcast(&globalBest, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Se un processo (diverso dal master) ha ottenuto la soluzione globale, lo invia al master.
    // Il master, se non ha la soluzione globale, attende la ricezione.
    if(mpi_rank != 0) {
        if(bestSolution == globalBest) {
            MPI_Send(bestColoring.data(), n, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    } else {
        // Il master controlla se la propria soluzione è globale, altrimenti riceve da uno dei lavoratori.
        vector<int> globalBestColoring;
        if(bestSolution == globalBest) {
            globalBestColoring = bestColoring;
        } else {
            globalBestColoring.resize(n);
            MPI_Status status;
            MPI_Recv(globalBestColoring.data(), n, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        }
        
        double wallTime = duration_cast<duration<double>>(steady_clock::now() - startTime).count();
        cout << "\nSoluzione migliore trovata: " << globalBest << " colori\n";
        cout << "Tempo totale: " << wallTime << " secondi\n";
        for (int i = 0; i < n; i++){
            cout << "Vertice " << i << " -> Colore " << globalBestColoring[i] << "\n";
        }
    }
    
    MPI_Finalize();
    return 0;
}