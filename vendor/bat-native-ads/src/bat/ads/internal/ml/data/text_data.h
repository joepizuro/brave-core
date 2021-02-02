/* Copyright (c) 2020 The Brave Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BAT_ADS_INTERNAL_ML_DATA_TEXT_DATA_H_  // NOLINT
#define BAT_ADS_INTERNAL_ML_DATA_TEXT_DATA_H_  // NOLINT

#include <map>
#include <string>
#include <vector>

#include "bat/ads/internal/ml/data/data.h"

namespace ads {
namespace ml {
namespace data {

class TextData : public Data {
 public:
  TextData();

  TextData(const TextData& text_data);

  TextData& operator=(const TextData& text_data);

  explicit TextData(const std::string& text);

  ~TextData() override;

  std::string GetText() const;

 private:
  std::string text_;
};

}  // namespace data
}  // namespace ml
}  // namespace ads

#endif  // BAT_ADS_INTERNAL_ML_DATA_TEXT_DATA_H_  // NOLINT
