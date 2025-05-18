// author:  Taichicchi
// created: 18.05.2025 19:00:00

#include <bits/stdc++.h>
using namespace std;
using ll = long long;
using namespace std::chrono;

constexpr int N = 36;
constexpr int M = 12;
constexpr int L = 1000000;

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
            for (int j = 0; j < (int)A[i].size(); ++j) cout << ' ' << A[i][j];
            cout << '\n';
        }
    }
};

struct TimeKeeper {
    double timeout;
    time_point<steady_clock> start_time;
    TimeKeeper(double timeout_ = 1.5) : timeout(timeout_), start_time(steady_clock::now()) {}
    double elapsed_time() const {
        return duration_cast<duration<double>>(steady_clock::now() - start_time).count();
    }
    bool is_timeout() const {
        return elapsed_time() > timeout;
    }
};

vector<int> make_row(int p, int index) {
    vector<int> result(M, 0);
    result[index] = p;
    int rest = 100 - p;
    int num_rest = M - 1;
    int base = rest / num_rest;
    int extra = rest % num_rest;
    int cur = 0;
    for (int i = 0; i < M; ++i) {
        if (i == index) continue;
        int add = base + (cur < extra ? 1 : 0);
        result[i] = add;
        cur++;
    }
    return result;
}

Output make_initial_solution(const Input& input, int p = 100) {
    string best_str = input.S[0];
    vector<char> C;
    vector<vector<int>> A;
    for (char ch : best_str) C.push_back(ch);
    while ((int)C.size() < M) C.push_back('a');
    for (int i = 0; i < M; ++i) {
        if (i < (int)best_str.size() - 1) {
            A.push_back(make_row(p, i + 1));
        } else if (i == (int)best_str.size() - 1) {
            A.push_back(make_row(p, 0));
        } else {
            A.push_back({9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8});
        }
    }
    return Output(C, A);
}

mt19937 rng(42);

Output get_neighbor(const Output& output) {
    auto C = output.C;
    auto A = output.A;
    uniform_int_distribution<int> dist_idx(0, M - 1);
    int index = dist_idx(rng);
    string chars = "abcdef";
    uniform_int_distribution<int> dist_char(0, (int)chars.size() - 1);
    C[index] = chars[dist_char(rng)];
    uniform_int_distribution<int> dist_d(1, 10);
    int d = dist_d(rng);
    int idx = dist_idx(rng);
    int idx1 = dist_idx(rng);
    int idx2 = dist_idx(rng);
    while (idx1 == idx2) idx2 = dist_idx(rng);
    int max_add = min({d, 100 - A[idx][idx1], A[idx][idx2]});
    if (max_add > 0) {
        A[idx][idx1] += max_add;
        A[idx][idx2] -= max_add;
    }
    return Output(C, A);
}

double compute_word_probability(const string& word, int L, const vector<char>& C, const vector<vector<int>>& A) {
    int M = (int)C.size();
    vector<char> w(word.begin(), word.end());
    int len_w = (int)w.size();
    map<pair<int, int>, int> states;
    int n = 0;
    for (int j = 0; j < M; ++j) {
        states[{0, j}] = n++;
        for (int i = 0; i < len_w - 1; ++i) {
            if (w[i] == C[j]) {
                states[{i + 1, j}] = n++;
            }
        }
    }
    vector<vector<double>> X(n, vector<double>(n, 0.0));
    for (auto& [key, j] : states) {
        int matchlen = key.first, u = key.second;
        for (int v = 0; v < M; ++v) {
            vector<char> next_;
            next_.insert(next_.end(), w.begin(), w.begin() + matchlen);
            next_.push_back(C[v]);
            int s = 0;
            while (s < (int)next_.size() && vector<char>(next_.begin() + s, next_.end()) != vector<char>(w.begin(), w.begin() + ((int)next_.size() - s))) {
                s++;
            }
            if ((int)next_.size() - s != len_w) {
                int i2 = states[{(int)next_.size() - s, v}];
                X[i2][j] += A[u][v] / 100.0;
            }
        }
    }
    int power = L - 1;
    vector<vector<double>> Y(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) Y[i][i] = 1.0;
    while (power) {
        if (power % 2) {
            vector<vector<double>> Z(n, vector<double>(n, 0.0));
            for (int i = 0; i < n; ++i)
                for (int k = 0; k < n; ++k)
                    for (int j = 0; j < n; ++j) Z[i][j] += X[i][k] * Y[k][j];
            Y = Z;
        }
        vector<vector<double>> Z(n, vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < n; ++k)
                for (int j = 0; j < n; ++j) Z[i][j] += X[i][k] * X[k][j];
        X = Z;
        power /= 2;
    }
    int init = (C[0] == w[0]) ? states[{1, 0}] : states[{0, 0}];
    double ret = 1.0;
    for (int i = 0; i < n; ++i) ret -= Y[i][init];
    return max(0.0, min(1.0, ret));
}

double compute_score(const Input& input, const Output& output, int L = L) {
    double total_score = 0.0;
    for (int i = 0; i < N; ++i) {
        double prob = compute_word_probability(input.S[i], L, output.C, output.A);
        total_score += input.P[i] * prob;
    }
    return round(total_score);
}

Output simulated_annealing(const Input& input, TimeKeeper& time_keeper) {
    double T0 = 1e4, T_end = 1e-2;
    int step = 0;
    Output best_output = make_initial_solution(input, 50);
    double best_score = compute_score(input, best_output);
    Output current_output = best_output;
    double current_score = best_score;
    while (true) {
        step++;
        double elapsed = time_keeper.elapsed_time();
        double ratio = min(elapsed / time_keeper.timeout, 1.0);
        double T_now = T0 * pow(T_end / T0, ratio);
        Output neighbor = get_neighbor(current_output);
        double neighbor_score = compute_score(input, neighbor);
        double delta = neighbor_score - current_score;
        if (delta > 0 || uniform_real_distribution<double>(0, 1)(rng) < exp(delta / T_now)) {
            cerr << "1 step: " << step << ", " << current_score << " -> " << neighbor_score << endl;
            current_output = neighbor;
            current_score = neighbor_score;
            if (current_score > best_score) {
                cerr << "2 step: " << step << " " << best_score << " -> " << current_score << endl;
                best_output = current_output;
                best_score = current_score;
            }
        }
        if (time_keeper.is_timeout()) break;
    }
    return best_output;
}

int main() {
    TimeKeeper time_keeper(1.9);
    string dummy;
    getline(cin, dummy);  // 1行目を読み飛ばす
    vector<string> S(N);
    vector<int> P(N);
    for (int i = 0; i < N; ++i) {
        string s;
        int p;
        cin >> s >> p;
        S[i] = s;
        P[i] = p;
    }
    Input input(S, P);
    Output output = simulated_annealing(input, time_keeper);
    output.print();
    double score = compute_score(input, output);
    cerr << "score " << score << endl;
    return 0;
}
