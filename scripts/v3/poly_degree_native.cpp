#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

void check_inputs(const torch::Tensor& p, const torch::Tensor& degree,
                  int64_t nmax) {
  TORCH_CHECK(p.device().is_cpu() && degree.device().is_cpu(),
              "degree-aware polynomial kernel is CPU-only");
  TORCH_CHECK(p.scalar_type() == torch::kFloat64,
              "degree-aware polynomial kernel requires float64 coefficients");
  TORCH_CHECK(degree.scalar_type() == torch::kInt64,
              "degree-aware polynomial kernel requires int64 degrees");
  TORCH_CHECK(p.is_contiguous() && degree.is_contiguous(),
              "degree-aware polynomial inputs must be contiguous");
  TORCH_CHECK(p.dim() == 4 && degree.dim() == 2,
              "expected coefficients[D,B,C,N+1] and degree[B,C]");
  TORCH_CHECK(degree.size(0) == p.size(1) && degree.size(1) == p.size(2),
              "degree shape does not match coefficient batch/category axes");
  TORCH_CHECK(nmax >= 0 && p.size(3) >= nmax + 1,
              "coefficient axis is shorter than nmax+1");
}

}  // namespace

std::vector<torch::Tensor> degree_product_forward(
    torch::Tensor coefficients, torch::Tensor degree, int64_t nmax) {
  coefficients = coefficients.contiguous();
  degree = degree.contiguous();
  check_inputs(coefficients, degree, nmax);

  const int64_t draws = coefficients.size(0);
  const int64_t batch = coefficients.size(1);
  const int64_t categories = coefficients.size(2);
  const int64_t stride = coefficients.size(3);
  const int64_t width = nmax + 1;
  auto prefix = torch::zeros(
      {draws, batch, categories + 1, width}, coefficients.options());

  const double* input = coefficients.data_ptr<double>();
  const int64_t* degrees = degree.data_ptr<int64_t>();
  double* states = prefix.data_ptr<double>();
  const int64_t sequences = draws * batch;

  at::parallel_for(0, sequences, 1, [&](int64_t begin, int64_t end) {
    for (int64_t sequence = begin; sequence < end; ++sequence) {
      const int64_t b = sequence % batch;
      double* first = states + sequence * (categories + 1) * width;
      first[0] = 1.0;
      for (int64_t c = 0; c < categories; ++c) {
        double* previous = first + c * width;
        double* next = previous + width;
        const double* polynomial =
            input + (sequence * categories + c) * stride;
        const int64_t d = std::clamp<int64_t>(
            degrees[b * categories + c], 0, nmax);
        for (int64_t n = 0; n <= nmax; ++n) {
          double value = 0.0;
          const int64_t upper = std::min<int64_t>(d, n);
          for (int64_t k = 0; k <= upper; ++k) {
            value += previous[n - k] * polynomial[k];
          }
          next[n] = value;
        }
      }
    }
  });

  return {prefix.select(2, categories).contiguous(), prefix};
}

std::vector<torch::Tensor> esp_forward(
    torch::Tensor weights, torch::Tensor lengths,
    int64_t nmax, int64_t block_size) {
  TORCH_CHECK(weights.device().is_cpu() && weights.scalar_type() == torch::kFloat64,
              "native ESP requires CPU float64 weights");
  TORCH_CHECK(weights.dim() >= 2 && nmax >= 0 && block_size > 0,
              "native ESP expects [..., rows, items], nmax>=0, block_size>0");
  weights = weights.contiguous();
  lengths = lengths.contiguous();
  TORCH_CHECK(lengths.device().is_cpu() && lengths.scalar_type() == torch::kInt64 &&
              lengths.dim() == 1 && lengths.size(0) == weights.size(-2),
              "native ESP lengths must be int64 with one entry per row");
  const int64_t items = weights.size(-1);
  const int64_t rows = weights.size(-2);
  const int64_t sequences = weights.numel() / std::max<int64_t>(items, 1);
  const int64_t width = nmax + 1;
  const int64_t blocks = (items + block_size - 1) / block_size;
  std::vector<int64_t> out_shape(weights.sizes().begin(), weights.sizes().end() - 1);
  out_shape.push_back(width);
  auto output = torch::zeros(out_shape, weights.options());
  auto boundary = torch::zeros({sequences, blocks + 1, width}, weights.options());
  const double* input = weights.data_ptr<double>();
  const int64_t* row_lengths = lengths.data_ptr<int64_t>();
  double* result = output.data_ptr<double>();
  double* saved = boundary.data_ptr<double>();

  at::parallel_for(0, sequences, 1, [&](int64_t begin, int64_t end) {
    std::vector<double> state(width, 0.0);
    for (int64_t sequence = begin; sequence < end; ++sequence) {
      std::fill(state.begin(), state.end(), 0.0);
      state[0] = 1.0;
      double* first_boundary = saved + sequence * (blocks + 1) * width;
      std::copy(state.begin(), state.end(), first_boundary);
      const double* row = input + sequence * items;
      const int64_t item_count = std::clamp<int64_t>(
          row_lengths[sequence % rows], 0, items);
      int64_t next_boundary = 1;
      for (int64_t item = 0; item < item_count; ++item) {
        const double value = row[item];
        const int64_t degree = std::min<int64_t>(nmax, item + 1);
        for (int64_t k = degree; k >= 1; --k) {
          state[k] += value * state[k - 1];
        }
        if ((item + 1) % block_size == 0 || item + 1 == item_count) {
          std::copy(state.begin(), state.end(),
                    first_boundary + next_boundary * width);
          ++next_boundary;
        }
      }
      std::copy(state.begin(), state.end(), result + sequence * width);
    }
  });
  return {output, boundary};
}

// Log-coordinate wrapper around the blocked subtraction-free ESP recursion.  The
// linear states are safe here because sparse_prepare centres every product weight at
// the trip maximum: 0 <= w_j <= 1, so an R=120 coefficient over at most 1,774 items is
// bounded by choose(1774, 120), whose logarithm is below float64 overflow.  Category
// attraction is applied later in log coordinates and is not part of this ESP.
torch::Tensor esp_backward(torch::Tensor grad_output, torch::Tensor weights,
                           torch::Tensor lengths, torch::Tensor boundary, int64_t nmax,
                           int64_t block_size) {
  grad_output = grad_output.contiguous();
  weights = weights.contiguous();
  lengths = lengths.contiguous();
  boundary = boundary.contiguous();
  TORCH_CHECK(weights.device().is_cpu() && weights.scalar_type() == torch::kFloat64,
              "native ESP requires CPU float64 weights");
  const int64_t items = weights.size(-1);
  const int64_t rows = weights.size(-2);
  const int64_t sequences = weights.numel() / std::max<int64_t>(items, 1);
  const int64_t width = nmax + 1;
  const int64_t blocks = (items + block_size - 1) / block_size;
  TORCH_CHECK(grad_output.numel() == sequences * width,
              "native ESP output-gradient shape mismatch");
  TORCH_CHECK(boundary.sizes() == torch::IntArrayRef({sequences, blocks + 1, width}),
              "native ESP boundary shape mismatch");
  auto grad_weights = torch::zeros_like(weights);
  const double* input = weights.data_ptr<double>();
  const int64_t* row_lengths = lengths.data_ptr<int64_t>();
  const double* output_grad = grad_output.data_ptr<double>();
  const double* saved = boundary.data_ptr<double>();
  double* input_grad = grad_weights.data_ptr<double>();

  at::parallel_for(0, sequences, 1, [&](int64_t begin, int64_t end) {
    std::vector<double> adjoint(width);
    std::vector<double> previous_adjoint(width);
    std::vector<double> local((block_size + 1) * width);
    for (int64_t sequence = begin; sequence < end; ++sequence) {
      std::copy(output_grad + sequence * width,
                output_grad + (sequence + 1) * width, adjoint.begin());
      const double* row = input + sequence * items;
      const double* first_boundary = saved + sequence * (blocks + 1) * width;
      double* row_grad = input_grad + sequence * items;
      const int64_t item_count = std::clamp<int64_t>(
          row_lengths[sequence % rows], 0, items);
      const int64_t used_blocks = (item_count + block_size - 1) / block_size;
      for (int64_t block = used_blocks - 1; block >= 0; --block) {
        const int64_t start = block * block_size;
        const int64_t finish = std::min<int64_t>(item_count, start + block_size);
        const int64_t length = finish - start;
        std::copy(first_boundary + block * width,
                  first_boundary + (block + 1) * width, local.begin());
        for (int64_t offset = 0; offset < length; ++offset) {
          double* previous = local.data() + offset * width;
          double* next = local.data() + (offset + 1) * width;
          std::copy(previous, previous + width, next);
          const double value = row[start + offset];
          const int64_t degree = std::min<int64_t>(nmax, start + offset + 1);
          for (int64_t k = degree; k >= 1; --k) {
            next[k] += value * previous[k - 1];
          }
        }
        for (int64_t offset = length - 1; offset >= 0; --offset) {
          const int64_t item = start + offset;
          const double* previous = local.data() + offset * width;
          const int64_t degree = std::min<int64_t>(nmax, item + 1);
          double value_grad = 0.0;
          for (int64_t k = 1; k <= degree; ++k) {
            value_grad += adjoint[k] * previous[k - 1];
          }
          row_grad[item] = value_grad;
          const double value = row[item];
          for (int64_t k = 0; k < nmax; ++k) {
            previous_adjoint[k] = adjoint[k] + value * adjoint[k + 1];
          }
          previous_adjoint[nmax] = adjoint[nmax];
          adjoint.swap(previous_adjoint);
        }
      }
    }
  });
  return grad_weights;
}

std::vector<torch::Tensor> esp_tree_forward(torch::Tensor weights, int64_t nmax) {
  TORCH_CHECK(weights.device().is_cpu() && weights.scalar_type() == torch::kFloat64 &&
              weights.dim() == 3 && nmax >= 0,
              "native ESP tree expects CPU float64 [draw,row,item] weights");
  weights = weights.contiguous();
  const int64_t draws = weights.size(0);
  const int64_t rows = weights.size(1);
  const int64_t items = weights.size(2);
  const int64_t sequences = draws * rows;
  auto leaves = torch::zeros({sequences, items, 2}, weights.options());
  const double* input = weights.data_ptr<double>();
  double* leaf = leaves.data_ptr<double>();
  at::parallel_for(0, sequences * items, 1, [&](int64_t begin, int64_t end) {
    for (int64_t q = begin; q < end; ++q) {
      leaf[2 * q] = 1.0;
      leaf[2 * q + 1] = input[q];
    }
  });
  std::vector<torch::Tensor> levels;
  levels.push_back(leaves);
  int64_t count = items;
  int64_t degree = 2;
  while (count > 1) {
    const int64_t parent_count = (count + 1) / 2;
    const int64_t parent_degree = std::min<int64_t>(2 * degree - 1, nmax + 1);
    auto parent = torch::zeros({sequences, parent_count, parent_degree}, weights.options());
    const double* child = levels.back().data_ptr<double>();
    double* output = parent.data_ptr<double>();
    at::parallel_for(0, sequences * parent_count, 1,
                     [&](int64_t begin, int64_t end) {
      for (int64_t task = begin; task < end; ++task) {
        const int64_t sequence = task / parent_count;
        const int64_t pair = task % parent_count;
        const int64_t left_id = 2 * pair;
        const int64_t right_id = left_id + 1;
        const double* left = child + (sequence * count + left_id) * degree;
        double* out = output + task * parent_degree;
        if (right_id >= count) {
          std::copy(left, left + std::min<int64_t>(degree, parent_degree), out);
          continue;
        }
        const double* right = child + (sequence * count + right_id) * degree;
        for (int64_t r = 0; r < degree && r < parent_degree; ++r) {
          const int64_t take = std::min<int64_t>(degree, parent_degree - r);
          for (int64_t a = 0; a < take; ++a) {
            out[r + a] += left[a] * right[r];
          }
        }
      }
    });
    levels.push_back(parent);
    count = parent_count;
    degree = parent_degree;
  }
  auto output = levels.back().select(1, 0).reshape({draws, rows, degree}).contiguous();
  if (degree < nmax + 1) {
    auto padded = torch::zeros({draws, rows, nmax + 1}, weights.options());
    padded.slice(-1, 0, degree).copy_(output);
    output = padded;
  }
  std::vector<torch::Tensor> result;
  result.push_back(output);
  result.insert(result.end(), levels.begin(), levels.end());
  return result;
}

torch::Tensor esp_tree_backward(torch::Tensor grad_output,
                                std::vector<torch::Tensor> levels,
                                int64_t original_items, int64_t nmax) {
  grad_output = grad_output.contiguous();
  TORCH_CHECK(!levels.empty() && original_items >= 0,
              "native ESP tree backward received no saved levels");
  const int64_t sequences = levels.front().size(0);
  const int64_t final_degree = levels.back().size(2);
  auto adjoint = torch::zeros_like(levels.back());
  adjoint.select(1, 0).copy_(grad_output.reshape({sequences, nmax + 1})
                            .slice(-1, 0, final_degree));

  for (int64_t level = static_cast<int64_t>(levels.size()) - 1; level >= 1; --level) {
    const auto child_tensor = levels[level - 1].contiguous();
    const auto parent_tensor = levels[level].contiguous();
    adjoint = adjoint.contiguous();
    const int64_t child_count = child_tensor.size(1);
    const int64_t child_degree = child_tensor.size(2);
    const int64_t parent_count = parent_tensor.size(1);
    const int64_t parent_degree = parent_tensor.size(2);
    auto child_adjoint = torch::zeros_like(child_tensor);
    const double* child = child_tensor.data_ptr<double>();
    const double* go_all = adjoint.data_ptr<double>();
    double* child_go = child_adjoint.data_ptr<double>();
    at::parallel_for(0, sequences * parent_count, 1,
                     [&](int64_t begin, int64_t end) {
      for (int64_t task = begin; task < end; ++task) {
        const int64_t sequence = task / parent_count;
        const int64_t pair = task % parent_count;
        const int64_t left_id = 2 * pair;
        const int64_t right_id = left_id + 1;
        const double* left = child + (sequence * child_count + left_id) * child_degree;
        double* dleft = child_go + (sequence * child_count + left_id) * child_degree;
        const double* go = go_all + task * parent_degree;
        if (right_id >= child_count) {
          std::copy(go, go + std::min<int64_t>(child_degree, parent_degree), dleft);
          continue;
        }
        const double* right = child +
            (sequence * child_count + right_id) * child_degree;
        double* dright = child_go +
            (sequence * child_count + right_id) * child_degree;
        for (int64_t r = 0; r < child_degree && r < parent_degree; ++r) {
          const int64_t take = std::min<int64_t>(child_degree, parent_degree - r);
          double right_gradient = 0.0;
          for (int64_t a = 0; a < take; ++a) {
            dleft[a] += go[r + a] * right[r];
            right_gradient += go[r + a] * left[a];
          }
          dright[r] = right_gradient;
        }
      }
    });
    adjoint = child_adjoint;
  }
  auto grad_weights = adjoint.select(2, 1)
      .slice(1, 0, original_items)
      .contiguous();
  return grad_weights;
}

// Log-coefficient interface to the same balanced ESP tree.  The ordinary adjoint of a
// positive polynomial can overflow even when the final derivative is a probability:
// d log(P_k)/d P_k = 1/P_k is formed first and only later multiplied by a tiny child
// coefficient.  Here h_k = P_k dL/dP_k is propagated instead.  For C_k=sum_ab A_a B_b,
//
//   h(A_a) += h(C_k) A_a B_b / C_k,
//
// and the multiplier is a conditional probability in [0,1].
std::vector<torch::Tensor> esp_tree_log_forward(torch::Tensor log_weights,
                                                int64_t nmax) {
  TORCH_CHECK(log_weights.device().is_cpu() &&
              log_weights.scalar_type() == torch::kFloat64 &&
              log_weights.dim() == 3,
              "native log-ESP tree expects CPU float64 [draw,row,item] log weights");
  auto packed = esp_tree_forward(torch::exp(log_weights.contiguous()), nmax);
  auto root = packed.front();
  auto log_root = torch::where(root > 0, torch::log(root),
                               torch::full_like(root, -std::numeric_limits<double>::infinity()));
  packed.front() = log_root;
  return packed;
}

torch::Tensor esp_tree_log_backward(torch::Tensor grad_log_output,
                                    std::vector<torch::Tensor> levels,
                                    int64_t original_items, int64_t nmax) {
  grad_log_output = grad_log_output.contiguous();
  TORCH_CHECK(!levels.empty() && original_items >= 0,
              "native log-ESP backward received no saved levels");
  const int64_t sequences = levels.front().size(0);
  const int64_t final_degree = levels.back().size(2);
  auto log_adjoint = torch::zeros_like(levels.back());
  log_adjoint.select(1, 0).copy_(
      grad_log_output.reshape({sequences, nmax + 1}).slice(-1, 0, final_degree));

  for (int64_t level = static_cast<int64_t>(levels.size()) - 1; level >= 1; --level) {
    const auto child_tensor = levels[level - 1].contiguous();
    const auto parent_tensor = levels[level].contiguous();
    log_adjoint = log_adjoint.contiguous();
    const int64_t child_count = child_tensor.size(1);
    const int64_t child_degree = child_tensor.size(2);
    const int64_t parent_count = parent_tensor.size(1);
    const int64_t parent_degree = parent_tensor.size(2);
    auto child_log_adjoint = torch::zeros_like(child_tensor);
    const double* child = child_tensor.data_ptr<double>();
    const double* parent = parent_tensor.data_ptr<double>();
    const double* hp_all = log_adjoint.data_ptr<double>();
    double* child_hp = child_log_adjoint.data_ptr<double>();
    at::parallel_for(0, sequences * parent_count, 1,
                     [&](int64_t begin, int64_t end) {
      for (int64_t task = begin; task < end; ++task) {
        const int64_t sequence = task / parent_count;
        const int64_t pair = task % parent_count;
        const int64_t left_id = 2 * pair;
        const int64_t right_id = left_id + 1;
        const double* left = child +
            (sequence * child_count + left_id) * child_degree;
        double* hleft = child_hp +
            (sequence * child_count + left_id) * child_degree;
        const double* hp = hp_all + task * parent_degree;
        if (right_id >= child_count) {
          std::copy(hp, hp + std::min<int64_t>(child_degree, parent_degree), hleft);
          continue;
        }
        const double* right = child +
            (sequence * child_count + right_id) * child_degree;
        double* hright = child_hp +
            (sequence * child_count + right_id) * child_degree;
        const double* out = parent + task * parent_degree;
        for (int64_t k = 0; k < parent_degree; ++k) {
          if (out[k] <= 0.0 || hp[k] == 0.0) continue;
          const double inverse = 1.0 / out[k];
          const bool ordinary = std::isfinite(inverse);
          const int64_t first_a = std::max<int64_t>(0, k - child_degree + 1);
          const int64_t last_a = std::min<int64_t>(child_degree - 1, k);
          for (int64_t a = first_a; a <= last_a; ++a) {
            const int64_t r = k - a;
            const double term = left[a] * right[r];
            const double fraction = ordinary ? term * inverse : term / out[k];
            const double contribution = hp[k] * fraction;
            hleft[a] += contribution;
            hright[r] += contribution;
          }
        }
      }
    });
    log_adjoint = child_log_adjoint;
  }
  return log_adjoint.select(2, 1).slice(1, 0, original_items).contiguous();
}

// Multiply category polynomials supplied as log coefficients and return log coefficients.
// Each input polynomial is divided by its largest coefficient before multiplication; the
// removed scalar is restored in log space.  This is algebraically exact and keeps every
// truncated convolution in a safe floating-point range.
std::vector<torch::Tensor> log_degree_product_forward(
    torch::Tensor log_coefficients, torch::Tensor degree, int64_t nmax) {
  log_coefficients = log_coefficients.contiguous();
  degree = degree.contiguous();
  check_inputs(log_coefficients, degree, nmax);
  const int64_t draws = log_coefficients.size(0);
  const int64_t batch = log_coefficients.size(1);
  const int64_t categories = log_coefficients.size(2);
  const int64_t stride = log_coefficients.size(3);
  const int64_t width = nmax + 1;
  const int64_t sequences = draws * batch;
  auto normalized = torch::zeros_like(log_coefficients);
  auto prefix = torch::zeros({draws, batch, categories + 1, width},
                             log_coefficients.options());
  auto log_output = torch::full({draws, batch, width},
      -std::numeric_limits<double>::infinity(), log_coefficients.options());
  const double* input = log_coefficients.data_ptr<double>();
  const int64_t* degrees = degree.data_ptr<int64_t>();
  double* q_all = normalized.data_ptr<double>();
  double* states = prefix.data_ptr<double>();
  double* result = log_output.data_ptr<double>();

  at::parallel_for(0, sequences, 1, [&](int64_t begin, int64_t end) {
    for (int64_t sequence = begin; sequence < end; ++sequence) {
      const int64_t b = sequence % batch;
      double* first = states + sequence * (categories + 1) * width;
      first[0] = 1.0;
      double log_scale = 0.0;
      for (int64_t c = 0; c < categories; ++c) {
        const int64_t d = std::clamp<int64_t>(degrees[b * categories + c], 0, nmax);
        const double* lp = input + (sequence * categories + c) * stride;
        double* q = q_all + (sequence * categories + c) * stride;
        double maximum = -std::numeric_limits<double>::infinity();
        for (int64_t k = 0; k <= d; ++k) maximum = std::max(maximum, lp[k]);
        TORCH_CHECK(std::isfinite(maximum), "category polynomial has no finite coefficient");
        for (int64_t k = 0; k <= d; ++k) q[k] = std::exp(lp[k] - maximum);
        log_scale += maximum;
        double* previous = first + c * width;
        double* next = previous + width;
        for (int64_t n = 0; n <= nmax; ++n) {
          double value = 0.0;
          const int64_t upper = std::min<int64_t>(d, n);
          for (int64_t k = 0; k <= upper; ++k) value += previous[n-k] * q[k];
          next[n] = value;
        }
      }
      const double* final_state = first + categories * width;
      double* out = result + sequence * width;
      for (int64_t n = 0; n <= nmax; ++n) {
        if (final_state[n] > 0.0) out[n] = std::log(final_state[n]) + log_scale;
      }
    }
  });
  return {log_output, normalized, prefix};
}

torch::Tensor log_degree_product_backward(
    torch::Tensor grad_log_output, torch::Tensor normalized,
    torch::Tensor degree, torch::Tensor prefix, int64_t nmax) {
  grad_log_output = grad_log_output.contiguous();
  normalized = normalized.contiguous();
  degree = degree.contiguous();
  prefix = prefix.contiguous();
  check_inputs(normalized, degree, nmax);
  const int64_t draws = normalized.size(0);
  const int64_t batch = normalized.size(1);
  const int64_t categories = normalized.size(2);
  const int64_t stride = normalized.size(3);
  const int64_t width = nmax + 1;
  const int64_t sequences = draws * batch;
  auto grad_log_coefficients = torch::zeros_like(normalized);
  const double* q_all = normalized.data_ptr<double>();
  const int64_t* degrees = degree.data_ptr<int64_t>();
  const double* states = prefix.data_ptr<double>();
  const double* output_grad = grad_log_output.data_ptr<double>();
  double* input_grad = grad_log_coefficients.data_ptr<double>();

  at::parallel_for(0, sequences, 1, [&](int64_t begin, int64_t end) {
    std::vector<double> hnext(width), hprevious(width);
    for (int64_t sequence = begin; sequence < end; ++sequence) {
      std::copy(output_grad + sequence * width,
                output_grad + (sequence + 1) * width, hnext.begin());
      const int64_t b = sequence % batch;
      const double* first = states + sequence * (categories + 1) * width;
      for (int64_t c = categories - 1; c >= 0; --c) {
        std::fill(hprevious.begin(), hprevious.end(), 0.0);
        const int64_t d = std::clamp<int64_t>(degrees[b * categories + c], 0, nmax);
        const double* previous = first + c * width;
        const double* next = previous + width;
        const double* q = q_all + (sequence * categories + c) * stride;
        double* hq = input_grad + (sequence * categories + c) * stride;
        for (int64_t n = 0; n <= nmax; ++n) {
          if (next[n] <= 0.0 || hnext[n] == 0.0) continue;
          const double inverse = 1.0 / next[n];
          const bool ordinary = std::isfinite(inverse);
          const int64_t upper = std::min<int64_t>(d, n);
          for (int64_t k = 0; k <= upper; ++k) {
            const int64_t m = n - k;
            const double term = previous[m] * q[k];
            const double fraction = ordinary ? term * inverse : term / next[n];
            const double contribution = hnext[n] * fraction;
            hprevious[m] += contribution;
            hq[k] += contribution;
          }
        }
        hnext.swap(hprevious);
      }
    }
  });
  return grad_log_coefficients;
}

torch::Tensor degree_product_backward(
    torch::Tensor grad_output, torch::Tensor coefficients,
    torch::Tensor degree, torch::Tensor prefix, int64_t nmax) {
  grad_output = grad_output.contiguous();
  coefficients = coefficients.contiguous();
  degree = degree.contiguous();
  prefix = prefix.contiguous();
  check_inputs(coefficients, degree, nmax);

  const int64_t draws = coefficients.size(0);
  const int64_t batch = coefficients.size(1);
  const int64_t categories = coefficients.size(2);
  const int64_t stride = coefficients.size(3);
  const int64_t width = nmax + 1;
  TORCH_CHECK(grad_output.sizes() == torch::IntArrayRef({draws, batch, width}),
              "gradient output has the wrong shape");
  TORCH_CHECK(prefix.sizes() ==
                  torch::IntArrayRef({draws, batch, categories + 1, width}),
              "saved prefix table has the wrong shape");

  auto grad_coefficients = torch::zeros_like(coefficients);
  const double* output_grad = grad_output.data_ptr<double>();
  const double* input = coefficients.data_ptr<double>();
  const int64_t* degrees = degree.data_ptr<int64_t>();
  const double* states = prefix.data_ptr<double>();
  double* input_grad = grad_coefficients.data_ptr<double>();
  const int64_t sequences = draws * batch;

  at::parallel_for(0, sequences, 1, [&](int64_t begin, int64_t end) {
    std::vector<double> adjoint(width);
    std::vector<double> previous_adjoint(width);
    for (int64_t sequence = begin; sequence < end; ++sequence) {
      const int64_t b = sequence % batch;
      const double* go = output_grad + sequence * width;
      std::copy(go, go + width, adjoint.begin());
      const double* first = states + sequence * (categories + 1) * width;
      for (int64_t c = categories - 1; c >= 0; --c) {
        const double* previous = first + c * width;
        const double* polynomial =
            input + (sequence * categories + c) * stride;
        double* polynomial_grad =
            input_grad + (sequence * categories + c) * stride;
        const int64_t d = std::clamp<int64_t>(
            degrees[b * categories + c], 0, nmax);

        for (int64_t k = 0; k <= d; ++k) {
          double value = 0.0;
          for (int64_t m = 0; m + k <= nmax; ++m) {
            value += adjoint[m + k] * previous[m];
          }
          polynomial_grad[k] = value;
        }
        std::fill(previous_adjoint.begin(), previous_adjoint.end(), 0.0);
        for (int64_t m = 0; m <= nmax; ++m) {
          double value = 0.0;
          const int64_t upper = std::min<int64_t>(d, nmax - m);
          for (int64_t k = 0; k <= upper; ++k) {
            value += adjoint[m + k] * polynomial[k];
          }
          previous_adjoint[m] = value;
        }
        adjoint.swap(previous_adjoint);
      }
    }
  });
  return grad_coefficients;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &degree_product_forward,
             "Degree-aware category polynomial forward");
  module.def("backward", &degree_product_backward,
             "Degree-aware category polynomial backward");
  module.def("esp_forward", &esp_forward,
             "Stable blocked elementary-symmetric-polynomial forward");
  module.def("esp_backward", &esp_backward,
             "Stable blocked elementary-symmetric-polynomial backward");
  module.def("esp_tree_forward", &esp_tree_forward,
             "Balanced elementary-symmetric-polynomial tree forward");
  module.def("esp_tree_backward", &esp_tree_backward,
             "Balanced elementary-symmetric-polynomial tree backward");
  module.def("esp_tree_log_forward", &esp_tree_log_forward,
             "Balanced ESP tree with log-coefficient output");
  module.def("esp_tree_log_backward", &esp_tree_log_backward,
             "Bounded logarithmic adjoint for the balanced ESP tree");
  module.def("log_degree_forward", &log_degree_product_forward,
             "Degree-aware category product in log-coefficient coordinates");
  module.def("log_degree_backward", &log_degree_product_backward,
             "Bounded logarithmic adjoint for the category product");
}
