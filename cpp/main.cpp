// author:  Taichicchi
// created: 18.05.2025 19:00:00

#include <bits/stdc++.h>
using namespace std;

constexpr int N = 36, M = 12, L = 1000000;

struct Input {
    vector<string> S;
    vector<int> P;
    Input(const vector<string>& S_, const vector<int>& P_) {
        vector<int> idxs(N);
        iota(idxs.begin(), idxs.end(), 0);
        sort(idxs.begin(), idxs.end(), [&](int i, int j) { return P_[i] > P_[j]; });
        S.resize(N);
        P.resize(N);
        for (int i = 0; i < N; ++i) {
            S[i] = S_[idxs[i]];
            P[i] = P_[idxs[i]];
        }
    }
};

struct Output {
    vector<char> C;
    vector<vector<int>> A;
    Output(const vector<char>& C_, const vector<vector<int>>& A_) : C(C_), A(A_) {}
    void print() const {
        for (int i = 0; i < (int)C.size(); ++i) {
            cout << C[i];
            for (int j = 0; j < (int)A[i].size(); ++j) {
                cout << " " << A[i][j];
            }
            cout << endl;
        }
    }
};

struct TimeKeeper {
    chrono::high_resolution_clock::time_point start_time;
    double timeout;
    TimeKeeper(double timeout_ = 1.5) : start_time(chrono::high_resolution_clock::now()), timeout(timeout_) {}
    double elapsed_time() const {
        auto now = chrono::high_resolution_clock::now();
        return chrono::duration<double>(now - start_time).count();
    }
    bool is_timeout() const {
        return elapsed_time() > timeout;
    }
};

Output make_initial_solution(const Input& input) {
    string best_str = input.S[0];
    vector<char> C;
    vector<vector<int>> A;
    for (char ch : best_str) C.push_back(ch);
    while ((int)C.size() < M) C.push_back('a');
    for (int i = 0; i < M; ++i) {
        vector<int> row(M, 5);
        if (i < (int)best_str.size() - 1) {
            row[i + 1] = 45;
        } else if (i == (int)best_str.size() - 1) {
            row[0] = 45;
        } else {
            row[i] = 45;
        }
        A.push_back(row);
    }
    return Output(C, A);
}

// Matrix exponentiation for double matrices
vector<vector<double>> mat_pow(vector<vector<double>> X, long long power) {
    int n = X.size();
    vector<vector<double>> Y(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) Y[i][i] = 1.0;
    while (power) {
        if (power & 1) {
            vector<vector<double>> Z(n, vector<double>(n, 0.0));
            for (int i = 0; i < n; ++i)
                for (int k = 0; k < n; ++k)
                    for (int j = 0; j < n; ++j)
                        Z[i][j] += X[i][k] * Y[k][j];
            swap(Y, Z);
        }
        vector<vector<double>> Z(n, vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < n; ++k)
                for (int j = 0; j < n; ++j)
                    Z[i][j] += X[i][k] * X[k][j];
        swap(X, Z);
        power >>= 1;
    }
    return Y;
}

double compute_word_probability(const string& word, int L, const vector<char>& C, const vector<vector<int>>& A) {
    int M = C.size();
    int len_w = word.size();
    map<pair<int, int>, int> states;
    int n = 0;
    for (int j = 0; j < M; ++j) {
        states[{0, j}] = n++;
        for (int i = 0; i < len_w - 1; ++i) {
            if (word[i] == C[j]) {
                states[{i + 1, j}] = n++;
            }
        }
    }
    vector<vector<double>> X(n, vector<double>(n, 0.0));
    for (auto& [key, j] : states) {
        int matchlen = key.first, u = key.second;
        for (int v = 0; v < M; ++v) {
            vector<char> next_;
            for (int i = 0; i < matchlen; ++i) next_.push_back(word[i]);
            next_.push_back(C[v]);
            int s = 0;
            while (s < (int)next_.size() && vector<char>(next_.begin() + s, next_.end()) != vector<char>(word.begin(), word.begin() + ((int)next_.size() - s))) {
                ++s;
            }
            if ((int)next_.size() - s != len_w) {
                int i2 = states[{(int)next_.size() - s, v}];
                X[i2][j] += A[u][v] / 100.0;
            }
        }
    }
    vector<vector<double>> Y = mat_pow(X, L - 1);
    int init = (C[0] == word[0]) ? states[{1, 0}] : states[{0, 0}];
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += Y[i][init];
    double ret = 1.0 - sum;
    return max(0.0, min(1.0, ret));
}

double compute_score(const Input& input, const Output& output) {
    double total_score = 0.0;
    for (int i = 0; i < N; ++i) {
        double prob = compute_word_probability(input.S[i], L, output.C, output.A);
        total_score += input.P[i] * prob;
    }
    return round(total_score);
}

Output simulated_annealing(const Input& input) {
    Output output = make_initial_solution(input);
    double best_score = compute_score(input, output);
    double now_score = best_score;
    Output best_output = output;
    Output now_output = output;

    double temperature = 1000.0;
    double cooling_rate = 0.99;
    TimeKeeper time_keeper(1.8);
    int it = 0;
    mt19937 rng(42);
    uniform_real_distribution<double> urd(0.0, 1.0);
    uniform_int_distribution<int> chardist(0, 5);
    string chars = "abcdef";
    while (!time_keeper.is_timeout()) {
        ++it;
        Output new_output = now_output;
        if (urd(rng) < 0.5) {
            int idx = rng() % M;
            char new_c = chars[chardist(rng)];
            while (new_c == new_output.C[idx]) new_c = chars[chardist(rng)];
            new_output.C[idx] = new_c;
        } else {
            int idx = rng() % M;
            int idx1 = rng() % M;
            int idx2 = rng() % M;
            while (idx1 == idx2) idx2 = rng() % M;
            int d = rng() % 10 + 1;
            d = min({d, 100 - new_output.A[idx][idx1], new_output.A[idx][idx2]});
            if (d > 0) {
                new_output.A[idx][idx1] += d;
                new_output.A[idx][idx2] -= d;
            } else {
                continue;
            }
        }
        double new_score = compute_score(input, new_output);
        if (new_score > now_score) {
            cerr << "it: " << it << ", " << now_score << " -> " << new_score << ", " << best_score << endl;
            now_score = new_score;
            now_output = new_output;
        } else {
            double delta = new_score - now_score;
            if (delta < 0 && urd(rng) < exp(delta / temperature)) {
                cerr << "it: " << it << ", " << now_score << " -> " << new_score << ", " << best_score << endl;
                now_score = new_score;
                now_output = new_output;
            }
        }
        if (now_score > best_score) {
            best_score = now_score;
            best_output = now_output;
        }
        temperature *= cooling_rate;
    }
    return best_output;
}

int main() {
    string dummy;
    getline(cin, dummy);  // skip first line
    vector<string> S;
    vector<int> P;
    for (int i = 0; i < N; ++i) {
        string s;
        int p;
        cin >> s >> p;
        S.push_back(s);
        P.push_back(p);
    }
    Input input(S, P);
    Output output = simulated_annealing(input);
    output.print();
    double score = compute_score(input, output);
    cerr << "score " << score << endl;
    return 0;
}