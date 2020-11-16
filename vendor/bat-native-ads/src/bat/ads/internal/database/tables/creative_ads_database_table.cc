/* Copyright (c) 2020 The Brave Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "bat/ads/internal/database/tables/creative_ads_database_table.h"

#include <utility>

#include "base/strings/stringprintf.h"
#include "bat/ads/internal/ads_impl.h"
#include "bat/ads/internal/database/database_statement_util.h"
#include "bat/ads/internal/database/database_table_util.h"
#include "bat/ads/internal/database/database_util.h"
#include "bat/ads/internal/logging.h"

namespace ads {
namespace database {
namespace table {

using std::placeholders::_1;

namespace {
const char kTableName[] = "creative_ads";
}  // namespace

CreativeAds::CreativeAds(
    AdsImpl* ads)
    : ads_(ads) {
  DCHECK(ads_);
}

CreativeAds::~CreativeAds() = default;

void CreativeAds::InsertOrUpdate(
    DBTransaction* transaction,
    const CreativeAdList& creative_ads) {
  DCHECK(transaction);

  if (creative_ads.empty()) {
    return;
  }

  DBCommandPtr command = DBCommand::New();
  command->type = DBCommand::Type::RUN;
  command->command = BuildInsertOrUpdateQuery(command.get(), creative_ads);

  transaction->commands.push_back(std::move(command));
}

void CreativeAds::Delete(
    ResultCallback callback) {
  DBTransactionPtr transaction = DBTransaction::New();

  util::Delete(transaction.get(), get_table_name());

  ads_->get_ads_client()->RunDBTransaction(std::move(transaction),
      std::bind(&OnResultCallback, _1, callback));
}

std::string CreativeAds::get_table_name() const {
  return kTableName;
}

void CreativeAds::Migrate(
    DBTransaction* transaction,
    const int to_version) {
  DCHECK(transaction);

  switch (to_version) {
    case 3: {
      MigrateToV3(transaction);
      break;
    }

    case 6: {
      MigrateToV6(transaction);
      break;
    }

    default: {
      break;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

int CreativeAds::BindParameters(
    DBCommand* command,
    const CreativeAdList& creative_ads) {
  DCHECK(command);

  int count = 0;

  int index = 0;
  for (const auto& creative_ad : creative_ads) {
    BindString(command, index++, creative_ad.creative_instance_id);
    BindBool(command, index++, creative_ad.conversion);
    BindInt(command, index++, creative_ad.per_day);
    BindInt(command, index++, creative_ad.total_max);
    BindString(command, index++, creative_ad.target_url);

    count++;
  }

  return count;
}

std::string CreativeAds::BuildInsertOrUpdateQuery(
    DBCommand* command,
    const CreativeAdList& creative_ads) {
  const int count = BindParameters(command, creative_ads);

  return base::StringPrintf(
      "INSERT OR REPLACE INTO %s "
          "(creative_instance_id, "
          "conversion, "
          "per_day, "
          "total_max, "
          "target_url) VALUES %s",
      get_table_name().c_str(),
      BuildBindingParameterPlaceholders(5, count).c_str());
}

void CreativeAds::CreateTableV3(
    DBTransaction* transaction) {
  DCHECK(transaction);

  const std::string query = base::StringPrintf(
      "CREATE TABLE %s "
          "(creative_set_id TEXT NOT NULL PRIMARY KEY UNIQUE "
              "ON CONFLICT REPLACE, "
          "conversion INTEGER NOT NULL DEFAULT 0, "
          "per_day INTEGER NOT NULL DEFAULT 0, "
          "total_max INTEGER NOT NULL DEFAULT 0, "
          "target_url TEXT NOT NULL)",
      get_table_name().c_str());

  DBCommandPtr command = DBCommand::New();
  command->type = DBCommand::Type::EXECUTE;
  command->command = query;

  transaction->commands.push_back(std::move(command));
}

void CreativeAds::CreateIndexV3(
    DBTransaction* transaction) {
  DCHECK(transaction);

  util::CreateIndex(transaction, get_table_name(), "creative_set_id");
}

void CreativeAds::MigrateToV3(
    DBTransaction* transaction) {
  DCHECK(transaction);

  util::Drop(transaction, get_table_name());

  CreateTableV3(transaction);
  CreateIndexV3(transaction);
}

void CreativeAds::CreateTableV6(
    DBTransaction* transaction) {
  DCHECK(transaction);

  const std::string query = base::StringPrintf(
      "CREATE TABLE %s "
          "(creative_instance_id TEXT NOT NULL PRIMARY KEY UNIQUE "
              "ON CONFLICT REPLACE, "
          "conversion INTEGER NOT NULL DEFAULT 0, "
          "per_day INTEGER NOT NULL DEFAULT 0, "
          "total_max INTEGER NOT NULL DEFAULT 0, "
          "target_url TEXT NOT NULL)",
      get_table_name().c_str());

  DBCommandPtr command = DBCommand::New();
  command->type = DBCommand::Type::EXECUTE;
  command->command = query;

  transaction->commands.push_back(std::move(command));
}

void CreativeAds::CreateIndexV6(
    DBTransaction* transaction) {
  DCHECK(transaction);

  util::CreateIndex(transaction, get_table_name(), "creative_instance_id");
}

void CreativeAds::MigrateToV6(
    DBTransaction* transaction) {
  DCHECK(transaction);

  util::Drop(transaction, get_table_name());

  CreateTableV6(transaction);
  CreateIndexV6(transaction);
}

}  // namespace table
}  // namespace database
}  // namespace ads
