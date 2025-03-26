// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "apply_xgrammar_token_bitmask/apply_xgrammar_token_bitmask.h"
#include "axpby/axpby.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Sample extension for MLX";

  m.def(
      "axpby",
      &my_ext::axpby,
      "x"_a,
      "y"_a,
      "alpha"_a,
      "beta"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Scale and sum two vectors element-wise
        ``z = alpha * x + beta * y``

        Follows numpy style broadcasting between ``x`` and ``y``
        Inputs are upcasted to floats if needed

        Args:
            x (array): Input array.
            y (array): Input array.
            alpha (float): Scaling factor for ``x``.
            beta (float): Scaling factor for ``y``.

        Returns:
            array: ``alpha * x + beta * y``
      )");

  m.def(
      "apply_xgrammar_token_bitmask",
      &my_ext::apply_xgrammar_token_bitmask,
      "bitmask"_a,
      "logits"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
          Apply a bitmask to vocabulary logits
          For each position, if the corresponding bit in bitmask is 1,
          keep the logit value; otherwise set to -inf

          Args:
              bitmask (array): Array of uint32 where each bit corresponds to a token
              logits (array): Array of float16 or float32 containing vocabulary logits
              stream (Stream, optional): Stream on which to schedule the operation

          Returns:
              array: Array with same shape and dtype as logits, where values are either
                    the original logit value (if corresponding bit is 1) or -inf
      )");
}
