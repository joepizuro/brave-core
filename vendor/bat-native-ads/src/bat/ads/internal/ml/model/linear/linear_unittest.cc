/* Copyright (c) 2020 The Brave Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at http://mozilla.org/MPL/2.0/. */

#include <cmath>
#include <vector>

#include "bat/ads/internal/ml/data/vector_data.h"
#include "bat/ads/internal/ml/model/linear/linear.h"

#include "bat/ads/internal/json_helper.h"
#include "bat/ads/internal/unittest_base.h"
#include "bat/ads/internal/unittest_util.h"

// npm run test -- brave_unit_tests --filter=BatAds*

namespace ads {
namespace ml {

class BatAdsLinearModelTest : public UnitTestBase {
 protected:
  BatAdsLinearModelTest() = default;

  ~BatAdsLinearModelTest() override = default;
};

TEST_F(BatAdsLinearModelTest, ThreeClassesPredictionTest) {
  // Arrange
  const std::map<std::string, VectorData> weights = {
      {"class_1", VectorData(std::vector<double>{1.0, 0.0, 0.0})},
      {"class_2", VectorData(std::vector<double>{0.0, 1.0, 0.0})},
      {"class_3", VectorData(std::vector<double>{0.0, 0.0, 1.0})}};

  const std::map<std::string, double> biases = {
      {"class_1", 0.0}, {"class_2", 0.0}, {"class_3", 0.0}};

  const model::Linear linear(weights, biases);
  const VectorData class1_data_vector(std::vector<double>{1.0, 0.0, 0.0});
  const VectorData class2_data_vector(std::vector<double>{0.0, 1.0, 0.0});
  const VectorData class3_data_vector(std::vector<double>{0.0, 1.0, 2.0});

  // Act
  const PredictionMap res1 = linear.Predict(class1_data_vector);
  const PredictionMap res2 = linear.Predict(class2_data_vector);
  const PredictionMap res3 = linear.Predict(class3_data_vector);

  // Assert
  ASSERT_TRUE(res1.at("class_1") > res1.at("class_2"));
  ASSERT_TRUE(res1.at("class_1") > res1.at("class_3"));

  ASSERT_TRUE(res2.at("class_2") > res2.at("class_1"));
  ASSERT_TRUE(res2.at("class_2") > res2.at("class_3"));

  EXPECT_TRUE(res3.at("class_3") > res3.at("class_1") &&
              res3.at("class_3") > res3.at("class_2"));
}

TEST_F(BatAdsLinearModelTest, BiasesPredictionTest) {
  // Arrange
  const std::map<std::string, VectorData> weights = {
      {"class_1", VectorData(std::vector<double>{1.0, 0.0, 0.0})},
      {"class_2", VectorData(std::vector<double>{0.0, 1.0, 0.0})},
      {"class_3", VectorData(std::vector<double>{0.0, 0.0, 1.0})}};

  const std::map<std::string, double> biases = {
      {"class_1", 0.5}, {"class_2", 0.25}, {"class_3", 1.0}};

  const model::Linear linear_biased(weights, biases);
  const VectorData avg_vector(std::vector<double>{1.0, 1.0, 1.0});

  // Act
  const PredictionMap res = linear_biased.Predict(avg_vector);

  // Assert
  EXPECT_TRUE(res.at("class_3") > res.at("class_1") &&
              res.at("class_3") > res.at("class_2") &&
              res.at("class_1") > res.at("class_2"));
}

TEST_F(BatAdsLinearModelTest, BinaryClassifierPredictionTest) {
  // Arrange
  const size_t kExpectedPredictionSize = 1;
  const std::map<std::string, VectorData> weights = {
      {"the_only_class", VectorData(std::vector<double>{0.3, 0.2, 0.25})},
  };

  const std::map<std::string, double> biases = {
      {"the_only_class", -0.45},
  };

  const model::Linear linear(weights, biases);
  const VectorData data_vector_0(std::vector<double>{1.07, 1.52, 0.91});
  const VectorData data_vector_1(std::vector<double>{1.11, 1.63, 1.21});

  // Act
  const PredictionMap res_0 = linear.Predict(data_vector_0);
  const PredictionMap res_1 = linear.Predict(data_vector_1);

  // Assert
  ASSERT_EQ(kExpectedPredictionSize, res_0.size());
  ASSERT_EQ(kExpectedPredictionSize, res_1.size());

  EXPECT_TRUE(res_0.at("the_only_class") < 0.5 &&
              res_1.at("the_only_class") > 0.5);
}

TEST_F(BatAdsLinearModelTest, TopPredictionsTest) {
  // Arrange
  const size_t kPredictionLimits[2] = {2, 1};
  const std::map<std::string, VectorData> weights = {
      {"class_1", VectorData(std::vector<double>{1.0, 0.5, 0.8})},
      {"class_2", VectorData(std::vector<double>{0.3, 1.0, 0.7})},
      {"class_3", VectorData(std::vector<double>{0.6, 0.9, 1.0})},
      {"class_4", VectorData(std::vector<double>{0.7, 1.0, 0.8})},
      {"class_5", VectorData(std::vector<double>{1.0, 0.2, 1.0})}};

  const std::map<std::string, double> biases = {{"class_1", 0.21},
                                                {"class_2", 0.22},
                                                {"class_3", 0.23},
                                                {"class_4", 0.22},
                                                {"class_5", 0.21}};

  const model::Linear linear_biased(weights, biases);
  const VectorData point_1(std::vector<double>{1.0, 0.99, 0.98, 0.97, 0.96});
  const VectorData point_2(std::vector<double>{0.83, 0.79, 0.91, 0.87, 0.82});
  const VectorData point_3(std::vector<double>{0.92, 0.95, 0.85, 0.91, 0.73});

  // Act
  const PredictionMap res_1 = linear_biased.GetTopPredictions(point_1);
  const PredictionMap res_2 =
      linear_biased.GetTopPredictions(point_2, kPredictionLimits[0]);
  const PredictionMap res_3 =
      linear_biased.GetTopPredictions(point_3, kPredictionLimits[1]);

  // Assert
  ASSERT_EQ(weights.size(), res_1.size());
  ASSERT_EQ(kPredictionLimits[0], res_2.size());
  EXPECT_EQ(kPredictionLimits[1], res_3.size());
}

}  // namespace ml
}  // namespace ads
