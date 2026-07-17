#include "omega/native_engine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr int kParameters = 4;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kHugePenalty = 1.0e18;
constexpr double kEps = 1.0e-12;

struct Dual {
    double value{};
    std::array<double, kParameters> derivative{};

    Dual() = default;
    explicit Dual(double v) : value(v) { derivative.fill(0.0); }

    static Dual variable(double v, int index) {
        Dual result(v);
        result.derivative.at(static_cast<size_t>(index)) = 1.0;
        return result;
    }
};

inline Dual operator+(const Dual& a, const Dual& b) {
    Dual result(a.value + b.value);
    for (int i = 0; i < kParameters; ++i) result.derivative[i] = a.derivative[i] + b.derivative[i];
    return result;
}
inline Dual operator+(const Dual& a, double b) { return a + Dual(b); }
inline Dual operator+(double a, const Dual& b) { return Dual(a) + b; }
inline Dual operator-(const Dual& a, const Dual& b) {
    Dual result(a.value - b.value);
    for (int i = 0; i < kParameters; ++i) result.derivative[i] = a.derivative[i] - b.derivative[i];
    return result;
}
inline Dual operator-(const Dual& a, double b) { return a - Dual(b); }
inline Dual operator-(double a, const Dual& b) { return Dual(a) - b; }
inline Dual operator-(const Dual& a) {
    Dual result(-a.value);
    for (int i = 0; i < kParameters; ++i) result.derivative[i] = -a.derivative[i];
    return result;
}
inline Dual operator*(const Dual& a, const Dual& b) {
    Dual result(a.value * b.value);
    for (int i = 0; i < kParameters; ++i) {
        result.derivative[i] = a.derivative[i] * b.value + a.value * b.derivative[i];
    }
    return result;
}
inline Dual operator*(const Dual& a, double b) { return a * Dual(b); }
inline Dual operator*(double a, const Dual& b) { return Dual(a) * b; }
inline Dual operator/(const Dual& a, const Dual& b) {
    const double denom = b.value * b.value;
    Dual result(a.value / b.value);
    for (int i = 0; i < kParameters; ++i) {
        result.derivative[i] = (a.derivative[i] * b.value - a.value * b.derivative[i]) / denom;
    }
    return result;
}
inline Dual operator/(const Dual& a, double b) { return a / Dual(b); }
inline Dual operator/(double a, const Dual& b) { return Dual(a) / b; }

inline Dual dual_exp(const Dual& x) {
    const double e = std::exp(x.value);
    Dual result(e);
    for (int i = 0; i < kParameters; ++i) result.derivative[i] = e * x.derivative[i];
    return result;
}
inline Dual dual_log(const Dual& x) {
    Dual result(std::log(x.value));
    for (int i = 0; i < kParameters; ++i) result.derivative[i] = x.derivative[i] / x.value;
    return result;
}
inline Dual dual_pow(const Dual& x, double exponent) {
    const double powered = std::pow(x.value, exponent);
    Dual result(powered);
    const double factor = exponent * std::pow(x.value, exponent - 1.0);
    for (int i = 0; i < kParameters; ++i) result.derivative[i] = factor * x.derivative[i];
    return result;
}
inline double scalar_value(double x) { return x; }
inline double scalar_value(const Dual& x) { return x.value; }
inline double scalar_exp(double x) { return std::exp(x); }
inline Dual scalar_exp(const Dual& x) { return dual_exp(x); }
inline double scalar_log(double x) { return std::log(x); }
inline Dual scalar_log(const Dual& x) { return dual_log(x); }
inline double scalar_pow(double x, double exponent) { return std::pow(x, exponent); }
inline Dual scalar_pow(const Dual& x, double exponent) { return dual_pow(x, exponent); }

template <typename Scalar>
Scalar clamp_constant(const Scalar& value, double low, double high) {
    if (scalar_value(value) < low) return Scalar(low);
    if (scalar_value(value) > high) return Scalar(high);
    return value;
}

template <typename Scalar>
Scalar positive_floor(const Scalar& value, double low) {
    return scalar_value(value) < low ? Scalar(low) : value;
}

template <typename Scalar>
Scalar production(const Scalar& biomass_input, const Scalar& k, const Scalar& r, int model, double pella_shape) {
    Scalar biomass = biomass_input;
    if (scalar_value(biomass) < kEps) biomass = Scalar(kEps);
    if (scalar_value(biomass) > 2.0 * scalar_value(k)) biomass = k * 2.0;
    Scalar value;
    if (model == OMEGA_FOX) {
        value = r * biomass * scalar_log(k / biomass);
    } else if (model == OMEGA_PELLA) {
        const double shape = std::max(pella_shape, 1.0e-6);
        value = r * biomass * (Scalar(1.0) - scalar_pow(biomass / k, shape)) / shape;
    } else {
        value = r * biomass * (Scalar(1.0) - biomass / k);
    }
    return scalar_value(value) < 0.0 ? Scalar(0.0) : value;
}

template <typename Scalar>
std::vector<Scalar> simulate(
    const double* catches,
    int n,
    const Scalar& k,
    const Scalar& r,
    const Scalar& initial_depletion,
    int model,
    double pella_shape
) {
    std::vector<Scalar> biomass(static_cast<size_t>(n));
    if (n <= 0) return biomass;
    biomass[0] = k * initial_depletion;
    for (int i = 1; i < n; ++i) {
        Scalar next = biomass[static_cast<size_t>(i - 1)]
                    + production(biomass[static_cast<size_t>(i - 1)], k, r, model, pella_shape)
                    - catches[i - 1];
        const double floor = std::max(1.0e-6 * scalar_value(k), kEps);
        if (scalar_value(next) < floor) next = Scalar(floor);
        biomass[static_cast<size_t>(i)] = next;
    }
    return biomass;
}

template <typename Scalar>
Scalar lognormal_nll(const double* observations, const std::vector<Scalar>& predictions, int n, const Scalar& sigma) {
    Scalar result(0.0);
    const double constant = 0.5 * std::log(2.0 * kPi);
    for (int i = 0; i < n; ++i) {
        const double observed = observations[i];
        const double predicted = scalar_value(predictions[static_cast<size_t>(i)]);
        if (!std::isfinite(observed) || observed <= 0.0 || !std::isfinite(predicted) || predicted <= 0.0) continue;
        Scalar residual = std::log(observed) - scalar_log(predictions[static_cast<size_t>(i)]);
        result = result + 0.5 * (residual / sigma) * (residual / sigma) + scalar_log(sigma) + constant;
    }
    return result;
}

template <typename Scalar>
Scalar evaluate_objective(
    const std::array<Scalar, kParameters>& theta,
    const double* catches,
    const double* index,
    const double* biomass_observed,
    int n,
    const OmegaProductionSettings& settings,
    std::vector<Scalar>* biomass_prediction,
    std::array<Scalar, OMEGA_COMPONENT_COUNT>* components
) {
    for (auto& component : *components) component = Scalar(0.0);

    const Scalar k = scalar_exp(theta[0]);
    const Scalar r = scalar_exp(theta[1]);
    const Scalar initial_depletion = Scalar(1.0) / (Scalar(1.0) + scalar_exp(-theta[2]));
    const Scalar sigma = clamp_constant(scalar_exp(theta[3]), 0.03, 1.5);

    double max_catch = 0.0;
    for (int i = 0; i < n; ++i) max_catch = std::max(max_catch, catches[i]);
    if (scalar_value(k) <= max_catch * 1.05 || scalar_value(r) <= 0.005 || scalar_value(r) > 1.2) {
        return Scalar(kHugePenalty);
    }

    *biomass_prediction = simulate(catches, n, k, r, initial_depletion, settings.model, settings.pella_shape);
    for (const auto& value : *biomass_prediction) {
        if (!std::isfinite(scalar_value(value))) return Scalar(kHugePenalty);
    }

    int index_count = 0;
    Scalar log_q_sum(0.0);
    for (int i = 0; i < n; ++i) {
        if (std::isfinite(index[i]) && index[i] > 0.0 && scalar_value((*biomass_prediction)[static_cast<size_t>(i)]) > 0.0) {
            log_q_sum = log_q_sum + std::log(index[i]) - scalar_log((*biomass_prediction)[static_cast<size_t>(i)]);
            ++index_count;
        }
    }
    if (index_count > 0) {
        const Scalar q = scalar_exp(log_q_sum / static_cast<double>(index_count));
        std::vector<Scalar> index_prediction(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) index_prediction[static_cast<size_t>(i)] = q * (*biomass_prediction)[static_cast<size_t>(i)];
        (*components)[OMEGA_INDEX_LIKELIHOOD] = std::max(settings.index_weight, 0.0) * lognormal_nll(index, index_prediction, n, sigma);
    }

    (*components)[OMEGA_BIOMASS_LIKELIHOOD] = std::max(settings.biomass_weight, 0.0)
        * lognormal_nll(biomass_observed, *biomass_prediction, n, scalar_value(sigma) < 0.10 ? Scalar(0.10) : sigma);

    if (settings.use_target_depletion && n > 0) {
        const Scalar predicted_depletion = positive_floor((*biomass_prediction)[static_cast<size_t>(n - 1)] / k, 1.0e-6);
        const double target = std::max(settings.target_depletion, 1.0e-6);
        const double sd = std::max(settings.target_depletion_cv, 0.05);
        const Scalar z = (scalar_log(predicted_depletion) - std::log(target)) / sd;
        (*components)[OMEGA_TERMINAL_DEPLETION] = 0.5 * z * z;
    }

    const double observation_prior_sd = std::max(settings.observation_prior_log_sd, 1.0e-6);
    const Scalar observation_z = (scalar_log(sigma) - std::log(std::max(settings.obs_cv, 0.03))) / observation_prior_sd;
    (*components)[OMEGA_OBSERVATION_PRIOR] = 0.5 * observation_z * observation_z;

    const double r_sd = std::max(std::sqrt(std::log(1.0 + settings.r_prior_cv * settings.r_prior_cv)), 1.0e-6);
    const Scalar productivity_z = (scalar_log(r) - std::log(std::max(settings.r_prior_median, 1.0e-6))) / r_sd;
    (*components)[OMEGA_PRODUCTIVITY_PRIOR] = 0.5 * productivity_z * productivity_z;

    const double initial_sd = std::max(settings.initial_depletion_prior_sd, 1.0e-6);
    const Scalar initial_z = (initial_depletion - settings.initial_depletion_prior_mean) / initial_sd;
    (*components)[OMEGA_INITIAL_DEPLETION_PRIOR] = std::max(settings.initial_depletion_prior_weight, 0.0) * initial_z * initial_z;

    const Scalar catch_ratio = max_catch / positive_floor(k, 1.0e-9);
    (*components)[OMEGA_CATCH_CAPACITY_PENALTY] = std::max(settings.catch_to_capacity_penalty_weight, 0.0) * catch_ratio * catch_ratio;

    Scalar objective(0.0);
    for (const auto& component : *components) objective = objective + component;
    return objective;
}

}  // namespace

extern "C" {

int omega_engine_abi_version(void) { return 3; }

const char* omega_engine_build_info(void) {
#ifdef _OPENMP
    static const std::string info = "Omega native engine 1.4.1; C++17; OpenMP=ON; AD=OmegaDual forward mode; ABI=3";
#else
    static const std::string info = "Omega native engine 1.4.1; C++17; OpenMP=OFF; AD=OmegaDual forward mode; ABI=3";
#endif
    return info.c_str();
}

int omega_engine_has_openmp(void) {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

int omega_engine_max_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void omega_engine_set_threads(int threads) {
#ifdef _OPENMP
    omp_set_num_threads(std::max(1, threads));
#else
    (void)threads;
#endif
}

double omega_production(double biomass, double k, double r, int model, double pella_shape) {
    return production(biomass, k, r, model, pella_shape);
}

int omega_simulate_production(
    const double* catches,
    int n,
    double k,
    double r,
    double initial_depletion,
    int model,
    double pella_shape,
    double* biomass_out
) {
    if (!catches || !biomass_out || n <= 0 || !std::isfinite(k) || k <= 0.0 || !std::isfinite(r) || r <= 0.0) return 1;
    const auto values = simulate(catches, n, k, r, initial_depletion, model, pella_shape);
    for (int i = 0; i < n; ++i) biomass_out[i] = values[static_cast<size_t>(i)];
    return 0;
}

int omega_production_objective_gradient(
    const double* theta,
    const int* years,
    const double* catches,
    const double* index,
    const double* biomass_observed,
    int n,
    const OmegaProductionSettings* settings,
    double* objective_out,
    double* gradient_out,
    double* biomass_prediction_out,
    double* components_out
) {
    (void)years;
    if (!theta || !catches || !index || !biomass_observed || n <= 0 || !settings || !objective_out) return 1;
    std::array<Dual, kParameters> variables;
    for (int i = 0; i < kParameters; ++i) variables[i] = Dual::variable(theta[i], i);
    std::vector<Dual> prediction;
    std::array<Dual, OMEGA_COMPONENT_COUNT> components;
    const Dual objective = evaluate_objective(variables, catches, index, biomass_observed, n, *settings, &prediction, &components);
    *objective_out = objective.value;
    if (gradient_out) {
        for (int i = 0; i < kParameters; ++i) gradient_out[i] = objective.derivative[i];
    }
    if (biomass_prediction_out) {
        for (int i = 0; i < n; ++i) biomass_prediction_out[i] = prediction.empty() ? 0.0 : prediction[static_cast<size_t>(i)].value;
    }
    if (components_out) {
        for (int i = 0; i < OMEGA_COMPONENT_COUNT; ++i) components_out[i] = components[static_cast<size_t>(i)].value;
    }
    return 0;
}

int omega_production_batch_objective(
    const double* theta_matrix,
    int candidates,
    const int* years,
    const double* catches,
    const double* index,
    const double* biomass_observed,
    int n,
    const OmegaProductionSettings* settings,
    double* objectives_out,
    double* gradients_out
) {
    if (!theta_matrix || candidates <= 0 || !catches || !index || !biomass_observed || n <= 0 || !settings || !objectives_out) return 1;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int candidate = 0; candidate < candidates; ++candidate) {
        const double* theta = theta_matrix + static_cast<size_t>(candidate) * kParameters;
        double* gradient = gradients_out ? gradients_out + static_cast<size_t>(candidate) * kParameters : nullptr;
        double objective = kHugePenalty;
        const int status = omega_production_objective_gradient(
            theta, years, catches, index, biomass_observed, n, settings,
            &objective, gradient, nullptr, nullptr
        );
        objectives_out[candidate] = status == 0 ? objective : kHugePenalty;
    }
    return 0;
}

}  // extern "C"
