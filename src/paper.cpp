// dsatur_pair_chiga.cpp
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

// Variabili globali per misurare il tempo e tenere traccia della migliore soluzione
steady_clock::time_point startTime;
bool searchCompleted = true;   // diventa false se si supera il time limit
int bestSolution = INF;        // numero minimo di colori trovato finora
vector<int> bestColoring;      // assegnamento migliore (di lunghezza n, per i vertici originali)

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
        return { (int)clique.size(), clique };
    }
    
    // -------------------------------
    // Heuristica per il coloring greedy (DSATURh) sul grafo (utile per calcolare un UB sul sottografo)
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
        return {numColors, color};
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
// Funzioni ausiliarie per lavorare con la colorazione parziale
// ----------------------------------------------------------------------

// Restituisce l'insieme degli indici dei vertici non ancora colorati
vector<int> getUncoloredVertices(const vector<int> &colors) {
    vector<int> uncolored;
    for (int i = 0; i < (int)colors.size(); i++) {
        if (colors[i] == -1)
            uncolored.push_back(i);
    }
    return uncolored;
}

// Dato un grafo g e un vettore di indici "vertices", restituisce il sottografo indotto su questi vertici
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

// Calcola i colori “feasible” per il vertice v (ossia, i colori non usati dai suoi adiacenti)
// Considera i colori già usati (0 ... currentColors-1) e aggiunge anche la possibilità di usare un nuovo colore (currentColors)
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
// Strutture e funzioni per il lower bound ausiliario (χGA)
// ----------------------------------------------------------------------

// Struttura per rappresentare un arco in un orientamento aciclico
struct Arc {
    int u, v;
};

// Data un grafo H, costruisce l'orientamento aciclico del complemento di H
// (si usa un ordinamento naturale 0,1,...,n-1 per garantire l'assenza di cicli)
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

// Costruisce il line–graph GA a partire dall’insieme di archi (ogni arco diventa un vertice in GA)
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

// Calcola il lower bound ausiliario basato su χGA.
// Dato il vettore di colorazione parziale, identifica l'insieme U dei vertici non colorati,
// estrae il sottografo H = G[U], costruisce GA e stima α(GA).
// Poi, secondo il teorema, LB_aux = currentColors + (|U| - α(GA)).
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
// Funzione φ per decidere se calcolare il lower bound ausiliario in un nodo "promettente".
// In questo esempio, utilizziamo la condizione: se il numero di vertici già colorati
// è compreso tra il 18% e il 35% del totale e se (bestSolution - currentColors) <= 10, allora φ = true.
bool phiFunction(int coloredCount, int totalVertices, int currentColors, int bestSolution) {
    if(coloredCount >= 0.18 * totalVertices && coloredCount <= 0.35 * totalVertices && (bestSolution - currentColors) <= 10)
         return true;
    return false;
}

// ----------------------------------------------------------------------
// DSATUR–χGA branching su coppia di vertici non adiacenti
//
// Utilizza una colorazione parziale (vettore "colors" di dimensione g.n, con -1 per i vertici non colorati)
// e il numero corrente di colori usati (currentColors). Il branching avviene scegliendo una coppia
// di vertici non adiacenti tra quelli non colorati e ramificando in due: 
//   - Branch 1: forzare la coppia ad avere lo stesso colore (assegnando un colore comune, fra quelli già usati o un nuovo colore)
//   - Branch 2: forzare la coppia ad avere colori differenti (provando tutte le assegnazioni ammissibili con c1 != c2)
// ----------------------------------------------------------------------
void dsaturPairRecursive(const Graph &g, vector<int> &colors, int currentColors, double timeLimit) {
    // Verifica time limit
    double elapsed = duration_cast<duration<double>>(steady_clock::now() - startTime).count();
    if(elapsed >= timeLimit) {
        searchCompleted = false;
        return;
    }
    
    int n = g.n;
    // Conta quanti vertici sono già colorati
    int coloredCount = 0;
    for (int i = 0; i < n; i++)
        if (colors[i] != -1)
            coloredCount++;
    
    // Caso base: se tutti i vertici sono colorati, aggiorna la migliore soluzione
    if(coloredCount == n) {
        if(currentColors < bestSolution) {
            bestSolution = currentColors;
            bestColoring = colors;
            cout << "Nuova soluzione migliore trovata con " << bestSolution << " colori.\n";
        }
        return;
    }
    
    // Calcola un lower bound classico: usa una clique euristica sul sottografo dei vertici non colorati
    vector<int> U = getUncoloredVertices(colors);
    Graph subG = getInducedSubgraph(g, U);
    auto cliquePair = subG.heuristicMaxClique();
    int cliqueSize = cliquePair.first;
    int lb = currentColors + cliqueSize;
    if(lb >= bestSolution) return; // potatura
    
    // Se il nodo è promettente, calcola il lower bound ausiliario
    if(phiFunction(coloredCount, n, currentColors, bestSolution)) {
        int auxLB = computeAuxiliaryLowerBound(g, colors, currentColors);
        if(auxLB >= bestSolution) return; // potatura
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
    // Se non esiste una coppia di vertici non adiacenti, allora i vertici non colorati formano una clique:
    // l'estensione migliore consiste nell'assegnare a ciascuno un nuovo colore distinto.
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
            cout << "Nodo completo: aggiornato bestSolution = " << bestSolution << "\n";
        }
        return;
    }
    
    // --- Branch 1: forzo v1 e v2 ad avere lo stesso colore ---
    vector<int> feas1 = feasibleColorsForVertex(v1, g, colors, currentColors);
    vector<int> feas2 = feasibleColorsForVertex(v2, g, colors, currentColors);
    // Calcola l'intersezione delle colorazioni ammissibili per entrambi
    vector<int> common;
    for (int c : feas1) {
         if(find(feas2.begin(), feas2.end(), c) != feas2.end())
             common.push_back(c);
    }
    for (int c : common) {
         int newCurrentColors = currentColors;
         if(c == currentColors)  // se si usa un nuovo colore, incremento il contatore
              newCurrentColors = currentColors + 1;
         int origV1 = colors[v1], origV2 = colors[v2];
         colors[v1] = c;
         colors[v2] = c;
         dsaturPairRecursive(g, colors, newCurrentColors, timeLimit);
         colors[v1] = origV1;
         colors[v2] = origV2;
         if(!searchCompleted) return;
    }
    
    // --- Branch 2: forzo v1 e v2 ad avere colori differenti ---
    for (int c1 : feas1) {
         for (int c2 : feas2) {
              if(c1 == c2) continue; // devono essere diversi
              int newCurrentColors = currentColors;
              if(c1 == currentColors || c2 == currentColors)
                  newCurrentColors = currentColors + 1;
              int origV1 = colors[v1], origV2 = colors[v2];
              colors[v1] = c1;
              colors[v2] = c2;
              dsaturPairRecursive(g, colors, newCurrentColors, timeLimit);
              colors[v1] = origV1;
              colors[v2] = origV2;
              if(!searchCompleted) return;
         }
    }
}

// ----------------------------------------------------------------------
// main()
// ----------------------------------------------------------------------
int main(int argc, char** argv) {
    if(argc < 3) {
        cerr << "Utilizzo: " << argv[0] << " <input_file> <time_limit_sec>\n";
        return 1;
    }
    string inputFile = argv[1];
    double timeLimit = atof(argv[2]);  // ad es. 10.0 secondi

    startTime = steady_clock::now();

    // Legge il grafo di input (formato COL)
    Graph root = readGraphFromCOLFile(inputFile);
    int n = root.n;
    
    // Inizializza la colorazione parziale: tutti i vertici non sono colorati (valore -1)
    vector<int> colors(n, -1);
    
    // Inizialmente nessun colore è stato usato
    int currentColors = 0;
    
    // Calcola un UB iniziale con una colorazione greedy (DSATURh) sul grafo intero
    auto coloringPair = root.heuristicColoring();
    bestSolution = coloringPair.first;
    bestColoring = coloringPair.second;
    cout << "Upper bound iniziale (DSATURh): " << bestSolution << "\n";
    
    // Avvia la ricerca ricorsiva
    dsaturPairRecursive(root, colors, currentColors, timeLimit);
    
    double wallTime = duration_cast<duration<double>>(steady_clock::now() - startTime).count();
    cout << "\nSoluzione migliore trovata: " << bestSolution << " colori\n";
    cout << "Tempo totale: " << wallTime << " secondi\n";
    
    // Output della colorazione (mapping: vertice -> colore)
    for (int i = 0; i < n; i++){
        cout << "Vertice " << i << " -> Colore " << bestColoring[i] << "\n";
    }
    
    return 0;
}