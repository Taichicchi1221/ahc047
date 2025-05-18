#include <bits/stdc++.h>
using namespace std;

constexpr int N = 36, M = 12, L = 1'000'000;

struct Input {
    vector<string> S;
    vector<int> P;
    Input(const vector<string>& S_, const vector<int>& P_) {
        vector<int> idxs(N);
        iota(idxs.begin(), idxs.end(), 0);
        vector<string> S2 = S_;
        vector<int> P2 = P_;
        sort(idxs.begin(), idxs.end(), [&](int i, int j) { return P2[i] > P2[j]; });
        S.resize(N);
        P.resize(N);
        for (int i = 0; i < N; ++i) {
            S[i] = S2[idxs[i]];
            P[i] = P2[idxs[i]];
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
    chrono::high_resolution_clock::time_point start;
    double timeout;
    TimeKeeper(double timeout_sec = 1.5) : start(chrono::high_resolution_clock::now()), timeout(timeout_sec) {}
    double elapsed_time() const {
        auto now = chrono::high_resolution_clock::now();
        return chrono::duration<double>(now - start).count();
    }
    bool is_timeout() const { return elapsed_time() > timeout; }
};

Output make_initial_solution(const Input& input, int p = 100) {
    string best_str = input.S[0];
    vector<char> C;
    vector<vector<int>> A;
    for (char ch : best_str) C.push_back(ch);
    while ((int)C.size() < M) C.push_back('a');
    auto make_row = [](int p, int index) {
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
            ++cur;
        }
        return result;
    };
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
    C[index] = chars[rng() % chars.size()];
    int d = uniform_int_distribution<int>(1, 10)(rng);
    int idx = dist_idx(rng);
    int idx1 = dist_idx(rng), idx2 = dist_idx(rng);
    while (idx1 == idx2) idx2 = dist_idx(rng);
    int max_add = min({d, 100 - A[idx][idx1], A[idx][idx2]});
    if (max_add > 0) {
        A[idx][idx1] += max_add;
        A[idx][idx2] -= max_add;
    }
    return Output(C, A);
}

double compute_word_probability(const string& word, int L, const vector<char>& C, const vector<vector<int>>& A) {
    int M = C.size();
    int len_w = word.size();
    map<pair<int, int>, int> states;
    int n = 0;
    for (int j = 0; j < M; ++j) {
        states[{0, j}] = n++;
        for (int i = 0; i < len_w - 1; ++i) {
            if (word[i] == C[j]) states[{i + 1, j}] = n++;
        }
    }
    vector<vector<double>> X(n, vector<double>(n, 0.0));
    for (auto& [key, j] : states) {
        int matchlen = key.first, u = key.second;
        for (int v = 0; v < M; ++v) {
            vector<char> next_;
            for (int k = 0; k < matchlen; ++k) next_.push_back(word[k]);
            next_.push_back(C[v]);
            int s = 0;
            while (vector<char>(next_.begin() + s, next_.end()) != vector<char>(word.begin(), word.begin() + next_.size() - s)) ++s;
            if ((int)next_.size() - s != len_w) {
                int i2 = states[{(int)next_.size() - s, v}];
                X[i2][j] += A[u][v] / 100.0;
            }
        }
    }
    vector<vector<double>> Y(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) Y[i][i] = 1.0;
    int power = L - 1;
    auto matmul = [](const vector<vector<double>>& a, const vector<vector<double>>& b) {
        int n = a.size();
        vector<vector<double>> c(n, vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < n; ++k)
                for (int j = 0; j < n; ++j) c[i][j] += a[i][k] * b[k][j];
        return c;
    };
    auto Xpow = X;
    while (power) {
        if (power & 1) Y = matmul(X, Y);
        X = matmul(X, X);
        power >>= 1;
    }
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

Output hill_climbing(const Input& input, TimeKeeper& time_keeper) {
    Output best_output = make_initial_solution(input, 50);
    double best_score = compute_score(input, best_output);
    while (true) {
        Output now_output = get_neighbor(best_output);
        double now_score = compute_score(input, now_output);
        if (now_score > best_score) {
            cerr << best_score << " -> " << now_score << endl;
            best_output = now_output;
            best_score = now_score;
        }
        if (time_keeper.is_timeout()) break;
    }
    return best_output;
}

int main() {
    TimeKeeper time_keeper(1.95);
    string dummy;
    getline(cin, dummy);  // 1行目を読み飛ばす
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
    Output output = hill_climbing(input, time_keeper);
    output.print();
    double score = compute_score(input, output);
    cerr << "score " << score << endl;
    return 0;
}