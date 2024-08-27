using namespace std;

#include <cmath>
#include <vector>
#include <iostream>
#include <mpi.h>
#include <fstream>
#include <fcntl.h>

const double PI = 3.14159265358979323846;

const int MAX_T = 5000;
const int T_RESOLUTION = 100;
const int T_NUM = MAX_T / T_RESOLUTION;
const string OUTPUT_FILE_PATH_WITHOUT_EXT = "output";
const string INPUT_FILE_PATH = "input.csv";

/// 応答解析のパラメータ
struct ResponseAccAnalyzerParams {
    uint32_t natural_period_ms; // 固有周期 [ms]
    uint32_t dt_ms;             // 入力データの時間分解能 [ms]
    double damping_h;           // 減衰定数
    double beta;                // ニューマークβ法のβ
    double init_x;              // 初期応答変位 [m]
    double init_v;              // 初期応答速度 [m/s]
    double init_a;              // 初期応答加速度 [gal]
    double init_xg;             // 初期地震動 [gal]
};

/// 1質点系の地震応答解析器
class ResponseAccAnalyzer {
public:
    /// パラメータをもとに応答解析器を生成する
    ResponseAccAnalyzer(const ResponseAccAnalyzerParams& params)
            : dt(static_cast<double>(params.dt_ms) / 1000.0),
              hardness(calc_hardness(100.0, params.natural_period_ms)),
              mass(100.0),
              damping_c(calc_damping_c(params.damping_h, 100.0, hardness)),
              beta(params.beta),
              init_x(params.init_x),
              init_v(params.init_v),
              init_a(params.init_a),
              init_xg(params.init_xg)
    {}

    /// この関数は結果に影響を与えません。（厳密には浮動小数点数の誤差があるため、影響があるかもしれません）
    /// この関数を使用すると質量に関するパラメータを変更できます。
    ResponseAccAnalyzer set_mass(double new_mass, uint32_t natural_period_ms, double damping_h) {
        mass = new_mass;
        hardness = calc_hardness(new_mass, natural_period_ms);
        damping_c = calc_damping_c(damping_h, new_mass, hardness);
        return *this;
    }

    /// 応答解析の結果
    struct Result {
        std::vector<double> x;       // 応答変位 [m]
        std::vector<double> v;       // 応答速度 [m/s]
        std::vector<double> a;       // 応答加速度 [gal]
        std::vector<double> abs_acc; // 絶対応答加速度 [gal]
    };

    /// 絶対応答加速度を計算する。
    /// xg: 地震の加速度波形 [gal]
    Result analyze(std::vector<double> xg) {
        // 初期地震動を挿入
        xg.insert(xg.begin(), init_xg);

        Result result;
        result.x.reserve(xg.size());
        result.v.reserve(xg.size());
        result.a.reserve(xg.size());
        result.abs_acc.reserve(xg.size());

        result.x.push_back(init_x);
        result.v.push_back(init_v);
        result.a.push_back(init_a);

        // 具体的な応答計算を行う
        //
        // x: 変位
        // v: 速度
        // a: 加速度
        // x_1: 次の変位
        // v_1: 次の速度
        // a_1: 次の加速度
        for (size_t i = 0; i < xg.size(); ++i) {
            double x = result.x[i];
            double v = result.v[i];
            double a = result.a[i];
            double xg_val = xg[i];

            double a_1_val = a_1(xg_val, a, v, x);
            double v_1_val = v_1(a, a_1_val, v);
            double x_1_val = x_1(a, a_1_val, v, x);

            result.x.push_back(x_1_val);
            result.v.push_back(v_1_val);
            result.a.push_back(a_1_val);
            result.abs_acc.push_back(abs_response_acc(a_1_val, xg_val));
        }

        return result;
    }

private:
    double dt;
    double hardness;
    double mass;
    double damping_c;
    double beta;
    double init_x;
    double init_v;
    double init_a;
    double init_xg;

    /// 質量と固有周期から剛性を計算する
    static double calc_hardness(double mass, uint32_t natural_period_ms) {
        return 4.0 * std::pow(PI, 2.0) * mass / std::pow(static_cast<double>(natural_period_ms) / 1000.0, 2.0);
    }

    /// 減衰定数を計算する
    static double calc_damping_c(double damping_h, double mass, double hardness) {
        return damping_h * 2.0 * std::sqrt(mass * hardness);
    }

    /// 絶対応答加速度
    double a_1(double xg, double a, double v, double x) const {
        double p_1 = -(xg * mass);
        return (p_1 - damping_c * (v + dt / 2.0 * a) - hardness * (x + dt * v + (1.0 / 2.0 - beta) * std::pow(dt, 2.0) * a)) /
               (mass + dt * damping_c / 2.0 + beta * std::pow(dt, 2.0) * hardness);
    }

    double v_1(double a, double a_1, double v) const {
        return v + (a_1 + a) * dt / 2.0;
    }

    double x_1(double a, double a_1, double v, double x) const {
        return x + v * dt + (1.0 / 2.0 - beta) * a * std::pow(dt, 2.0) + beta * a_1 * std::pow(dt, 2.0);
    }

    /// 応答加速度と地震動から絶対応答加速度を計算する
    static double abs_response_acc(double a, double xg) {
        return a + xg;
    }
};

int main(int argc, char *argv[]) {
    int size = 1, rank = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // sizeはT_NUMの約数でなければならない
    if (T_NUM % size != 0) {
        cerr << "size must be a multiple of 50" << endl;
        return 1;
    }


    // 地震波データ
    vector<double> seismic_data;
    ifstream file(INPUT_FILE_PATH);
    if(file.is_open()){
        string line;
        while (getline(file, line)) {
            seismic_data.push_back(stod(line));
            try{
                double num = stod(line);
                seismic_data.push_back(num);
            } catch (const invalid_argument& e) {
                cerr << "Invalid argument: " << e.what() << endl;
            } catch (const out_of_range& e) {
                cerr << "Out of range: " << e.what() << endl;
            }
        }
        file.close();
    }else{
        cerr << "Failed to open input file" << endl;
    }

    // 絶対応答加速度スペクトル
    vector<double> response_spectrum;
    response_spectrum.reserve(T_NUM);

    uint32_t start = rank * T_NUM * T_RESOLUTION / size;
    uint32_t end = (rank + 1) * T_NUM * T_RESOLUTION / size;

    // 固有周期ごとに最大絶対応答加速度を求める
    for (uint32_t T = start; T < end; T+=T_RESOLUTION) {
        // パラメータを設定してResponseAccAnalyzerを初期化
        ResponseAccAnalyzerParams params = {
                .natural_period_ms = T, // 固有周期 [ms]
                .dt_ms = 10,              // 入力データの時間分解能 [ms]
                .damping_h = 0.05,        // 減衰定数
                .beta = 0.25,             // ニューマークβ法のβ
                .init_x = 0.0,            // 初期応答変位 [m]
                .init_v = 0.0,            // 初期応答速度 [m/s]
                .init_a = 0.0,            // 初期応答加速度 [gal]
                .init_xg = 0.0            // 初期地震動 [gal]
        };

        ResponseAccAnalyzer analyzer(params);

        // 解析を実行
        auto result = analyzer.analyze(seismic_data);

        // abs_accの最大値を求める
        double max_abs_acc = 0;
        for (size_t i = 0; i < result.abs_acc.size(); ++i) {
            // 絶対応答加速度の最大値は絶対値から求める。
            double abs_abs_acc = abs(result.abs_acc[i]);

            if (abs_abs_acc > max_abs_acc){
                max_abs_acc = abs_abs_acc;
            }
        }

        response_spectrum.push_back(max_abs_acc);
    }


    // ファイルを出力
    // rankごとにファイルを作成して書き出す
    ofstream output_file(OUTPUT_FILE_PATH_WITHOUT_EXT + "-" + to_string(rank) + ".csv");
    if(output_file.is_open()){
        for (size_t i = 0; i < response_spectrum.size(); ++i) {
            output_file << response_spectrum[i] << endl;
        }
        output_file.close();
    }else{
        cerr << "Failed to open file" << endl;
    }

    return 0;
}
