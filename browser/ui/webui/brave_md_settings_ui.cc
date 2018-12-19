/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "brave/browser/ui/webui/brave_md_settings_ui.h"

#include "base/command_line.h"
#include "brave/browser/resources/grit/brave_settings_resources.h"
#include "brave/browser/resources/grit/brave_settings_resources_map.h"
#include "brave/browser/ui/webui/settings/brave_appearance_handler.h"
#include "brave/browser/ui/webui/settings/brave_privacy_handler.h"
#include "brave/browser/ui/webui/settings/default_brave_shields_handler.h"
#include "brave/common/brave_switches.h"
#include "chrome/browser/profiles/profile.h"
#include "chrome/browser/ui/webui/settings/metrics_reporting_handler.h"
#include "content/public/browser/web_ui_data_source.h"

#if defined(OS_MACOSX)
#include "brave/browser/ui/webui/settings/brave_relaunch_handler_mac.h"
#endif

BraveMdSettingsUI::BraveMdSettingsUI(content::WebUI* web_ui,
                                     const std::string& host)
    : MdSettingsUI(web_ui) {
  web_ui->AddMessageHandler(std::make_unique<settings::MetricsReportingHandler>());
  web_ui->AddMessageHandler(std::make_unique<BraveAppearanceHandler>());
  web_ui->AddMessageHandler(std::make_unique<BravePrivacyHandler>());
  web_ui->AddMessageHandler(std::make_unique<DefaultBraveShieldsHandler>());

  #if defined(OS_MACOSX)
  // Use sparkle's relaunch api for browser relaunch on update.
  web_ui->AddMessageHandler(std::make_unique<BraveRelaunchHandler>());
  #endif
}

BraveMdSettingsUI::~BraveMdSettingsUI() {
}

// static
void BraveMdSettingsUI::AddResources(content::WebUIDataSource* html_source,
                                Profile* profile) {
  for (size_t i = 0; i < kBraveSettingsResourcesSize; ++i) {
    html_source->AddResourcePath(kBraveSettingsResources[i].name,
                                 kBraveSettingsResources[i].value);
  }

  const base::CommandLine& command_line =
      *base::CommandLine::ForCurrentProcess();
  html_source->AddBoolean("isSyncEnabled",
                          command_line.HasSwitch(switches::kEnableBraveSync));
}
