/* Copyright (c) 2020 The Brave Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at http://mozilla.org/MPL/2.0/. */

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>

#include "base/notreached.h"
#include "bat/ads/internal/ml/ml_util.h"

namespace ads {
namespace ml {

PredictionMap Softmax(const PredictionMap& y) {
  double maximum = -std::numeric_limits<double>::infinity();
  for (const auto& x : y) {
    maximum = std::max(maximum, x.second);
  }
  std::map<std::string, double> rtn;
  double sum_exp = 0.0;
  for (const auto& x : y) {
    double val = std::exp(x.second - maximum);
    rtn[x.first] = val;
    sum_exp += val;
  }
  for (const auto& x : rtn) {
    rtn[x.first] /= sum_exp;
  }
  return rtn;
}

TransformationPtr GetTransformationCopy(const TransformationPtr& tr_ptr) {
  if (tr_ptr->GetType() == transformation::TransformationType::LOWERCASE) {
    transformation::Lowercase* lowercase_ptr =
        static_cast<transformation::Lowercase*>(tr_ptr.get());
    transformation::Lowercase lowercase_copy = *lowercase_ptr;
    return std::make_unique<transformation::Lowercase>(lowercase_copy);
  }

  if (tr_ptr->GetType() == transformation::TransformationType::HASHED_NGRAMS) {
    transformation::HashedNGrams* hashed_n_grams_ptr =
        static_cast<transformation::HashedNGrams*>(tr_ptr.get());
    transformation::HashedNGrams hashed_n_grams_ptr_copy = *hashed_n_grams_ptr;
    return std::make_unique<transformation::HashedNGrams>(
        hashed_n_grams_ptr_copy);
  }

  if (tr_ptr->GetType() == transformation::TransformationType::NORMALIZATION) {
    transformation::Normalization* normalization_ptr =
        static_cast<transformation::Normalization*>(tr_ptr.get());
    transformation::Normalization normalization_copy = *normalization_ptr;
    return std::make_unique<transformation::Normalization>(normalization_copy);
  }

  NOTREACHED();
  return TransformationPtr(nullptr);
}

TransformationVector GetTransformationVectorCopy(
    const TransformationVector& tr_vect) {
  TransformationVector tr_vect_copy;
  size_t transformation_count = tr_vect.size();
  for (size_t i = 0; i < transformation_count; ++i) {
    tr_vect_copy.push_back(GetTransformationCopy(tr_vect[i]));
  }
  return tr_vect_copy;
}

}  // namespace ml
}  // namespace ads
