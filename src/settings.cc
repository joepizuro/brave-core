/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include <stdlib.h>

#include "settings.h"
#include "json_helper.h"

namespace ads {

Settings::Settings() : settings_state_(new SETTINGS_STATE()) {}

Settings::~Settings() = default;

bool Settings::FromJson(const std::string& json) {
  SETTINGS_STATE state;
  if (!LoadFromJson(state, json.c_str())) {
    return false;
  }

  settings_state_.reset(new SETTINGS_STATE(state));

  return true;
}

bool Settings::IsAdsEnabled() const {
  return settings_state_->ads_enabled;
}

const std::string Settings::GetAdsLocale() const {
  return settings_state_->ads_locale;
}

uint64_t Settings::GetAdsPerHour() const {
  return std::strtoull(settings_state_->ads_per_hour.c_str(), nullptr, 10);
}

uint64_t Settings::GetAdsPerDay() const {
  return std::strtoull(settings_state_->ads_per_day.c_str(), nullptr, 10);
}

}  // namespace ads
