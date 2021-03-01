/* Copyright (c) 2020 The Brave Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at http://mozilla.org/MPL/2.0/. */

#include <string>
#include <vector>

#include "bat/ads/internal/ml/data/text_data.h"
#include "bat/ads/internal/ml/data/vector_data.h"
#include "bat/ads/internal/ml/transformation/hashed_ngrams.h"

#include "bat/ads/internal/unittest_base.h"
#include "bat/ads/internal/unittest_util.h"

// npm run test -- brave_unit_tests --filter=BatAds*

namespace ads {
namespace ml {

class BatAdsHashedNGramsTest : public UnitTestBase {
 protected:
  BatAdsHashedNGramsTest() = default;

  ~BatAdsHashedNGramsTest() override = default;
};

TEST_F(BatAdsHashedNGramsTest, HashingTest) {
  // Arrange
  const int kDefaultBucketCount = 10000;
  const size_t kExpectedElementCount = 10;
  const std::string kTestString = "tiny";
  const std::unique_ptr<Data> text_data =
      std::make_unique<TextData>(TextData(kTestString));

  transformation::HashedNGrams hashed_ngrams;

  // Act
  const std::unique_ptr<Data> hashed_data = hashed_ngrams.Apply(text_data);

  ASSERT_EQ(hashed_data->GetType(), DataType::VECTOR_DATA);

  VectorData* hashed_vect_data = static_cast<VectorData*>(hashed_data.get());

  // Assert
  // 10000 is the default size
  ASSERT_EQ(hashed_vect_data->GetDimensionCount(), kDefaultBucketCount);

  // Hashes for [t, i, n, y, ti, in, ny, tin, iny, tiny] -- 10 in total
  EXPECT_EQ(hashed_vect_data->GetRawData().size(), kExpectedElementCount);
}

TEST_F(BatAdsHashedNGramsTest, CustomHashingTest) {
  // Arrange
  const size_t kExpectedElementCount = 3;
  const std::string kTestString = "tiny";
  const std::unique_ptr<Data> text_data =
      std::make_unique<TextData>(TextData(kTestString));

  transformation::HashedNGrams hashed_ngrams(3, std::vector<int>{1, 2, 3});

  // Act
  const std::unique_ptr<Data> hashed_data = hashed_ngrams.Apply(text_data);

  ASSERT_EQ(hashed_data->GetType(), DataType::VECTOR_DATA);

  VectorData* hashed_vect_data = static_cast<VectorData*>(hashed_data.get());

  // Assert
  ASSERT_EQ(hashed_vect_data->GetDimensionCount(), 3);

  EXPECT_EQ(hashed_vect_data->GetRawData().size(), kExpectedElementCount);
}

}  // namespace ml
}  // namespace ads
